import os

from mantid.simpleapi import mtd
from mantid import config

import numpy as np

from garnet.plots.base import Pages
from garnet.plots.volume import SlicePlot
from garnet.reduction.data import DataModel
from garnet.reduction.crystallography import space_point, point_laue
from garnet.config.instruments import beamlines

config["Q.convention"] = "Crystallography"


class Normalization:
    def __init__(self, plan):
        self.plan = plan
        self.params = plan["Normalization"]

        self.validate_params()

    def validate_params(self):
        symbols = list(space_point.keys()) + list(point_laue.keys())

        if self.params.get("Symmetry") is not None:
            symmetry = self.params["Symmetry"].replace(" ", "")
            assert symmetry in symbols
            if space_point.get(symmetry) is not None:
                symmetry = space_point[symmetry]
            symmetry = point_laue[symmetry]
            self.params["Symmetry"] = symmetry

        assert len(self.params["Projections"]) == 3
        assert np.abs(np.linalg.det(self.params["Projections"])) > 0

        assert len(self.params["Bins"]) == 3
        assert all([isinstance(val, int) for val in self.params["Bins"]])
        assert (np.array(self.params["Bins"]) > 0).all()
        assert np.prod(self.params["Bins"]) < 1001**3  # memory usage limit

        assert len(self.params["Extents"]) == 3
        assert (np.diff(self.params["Extents"], axis=1) >= 0).all()

    @staticmethod
    def normalize_parallel(plan, runs, proc):
        plan["Runs"] = runs
        plan["OutputName"] += "_p{}".format(proc)

        instance = Normalization(plan)

        return instance.normalize()

    def normalize(self):
        output_file = self.get_output_file()
        diag_file = self.get_diagnostic_file()

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        runs = self.plan["Runs"]

        if data.laue:
            grouping_file = diag_file.replace(".nxs", ".xml")

            data.preprocess_detectors()
            data.create_grouping(grouping_file, self.plan.get("Grouping"))
            mtd.remove("detectors")

            for run in runs:
                data.load_data("data", self.plan["IPTS"], run)

                data.load_generate_normalization(
                    self.plan["VanadiumFile"], self.plan.get("FluxFile")
                )

                data.apply_calibration(
                    "data",
                    self.plan.get("DetectorCalibration"),
                    self.plan.get("TubeCalibration"),
                )

                data.apply_mask("data", self.plan.get("MaskFile"))

                data.crop_for_normalization("data")

                data.load_background(self.plan.get("BackgroundFile"), "data")

                data.group_pixels(grouping_file, "data")

                data.load_clear_UB(self.plan["UBFile"], "data")

                data.convert_to_Q_sample("data", "md", lorentz_corr=False)

                data.normalize_to_hkl(
                    "md",
                    self.params["Projections"],
                    self.params["Extents"],
                    self.params["Bins"],
                    symmetry=self.params.get("Symmetry"),
                )
        else:
            if self.plan["Instrument"] == "WAND²":
                data.load_data("md", self.plan["IPTS"], runs, self.plan.get("Grouping"))

                data.load_generate_normalization(self.plan["VanadiumFile"])

                if self.plan["UBFile"] is not None:
                    data.load_clear_UB(self.plan["UBFile"], "md")

                data.load_background(self.plan.get("BackgroundFile"), "md")

                data.normalize_to_hkl(
                    "md",
                    self.params["Projections"],
                    self.params["Extents"],
                    self.params["Bins"],
                    symmetry=self.params.get("Symmetry"),
                )

            else:
                for run in runs:
                    data.load_data(
                        "md", self.plan["IPTS"], run, self.plan.get("Grouping")
                    )

                    data.load_generate_normalization(self.plan["VanadiumFile"])

                    if self.plan["UBFile"] is not None:
                        data.load_clear_UB(self.plan["UBFile"], "md")

                    data.load_background(self.plan.get("BackgroundFile"), "md")

                    data.normalize_to_hkl(
                        "md",
                        self.params["Projections"],
                        self.params["Extents"],
                        self.params["Bins"],
                        symmetry=self.params.get("Symmetry"),
                    )

        UB_file = output_file.replace(".nxs", ".mat")
        data.save_UB(UB_file, "md")

        data_file = self.get_file(output_file, "data")
        norm_file = self.get_file(output_file, "norm")

        data.save_histograms(data_file, "md_data")
        data.save_histograms(norm_file, "md_norm")

        if mtd.doesExist("md_bkg_data") and mtd.doesExist("md_bkg_norm"):
            data_file = self.get_file(output_file, "bkg_data")
            norm_file = self.get_file(output_file, "bkg_norm")

            data.save_histograms(data_file, "md_bkg_data")
            data.save_histograms(norm_file, "md_bkg_norm")

        mtd.clear()

        os.remove(grouping_file)

        return output_file

    def get_file(self, file, ws=""):
        """
        Update filename with identifier name and optional workspace name.

        Parameters
        ----------
        file : str
            Original file name.
        ws : str, optional
            Name of workspace. The default is ''.

        Returns
        -------
        output_file : str
            File with updated name for identifier and workspace name.

        """

        if len(ws) > 0:
            ws = "_" + ws

        return self.append_name(file).replace(".nxs", ws + ".nxs")

    def append_name(self, file):
        """
        Update filename with identifier name

        Parameters
        ----------
        file : str
            Original file name.

        Returns
        -------
        output_file : str
            File with updated name for identifier name.

        """

        append = (
            self.projection_name()
            + self.extents_name()
            + self.binning_name()
            + self.symmetry_name()
        )

        name, ext = os.path.splitext(file)

        return name + append + ext

    def extents_name(self):
        """
        Min/max pairs for each dimensional extents.

        `_[min_0,max_0]_[min_1,max_1]_[min_2,max_2]`

        Returns
        -------
        extents : str
            Underscore separated list.

        """

        extents = self.params.get("Extents")

        return "".join(["_[{},{}]".format(*extent) for extent in extents])

    def binning_name(self):
        """
        Bin size for each dimension.

        `_N0xN1xN2`

        Returns
        -------
        bins : str
            Cross separated integers.

        """

        bins = self.params.get("Bins")

        return "_" + "x".join(np.array(bins).astype(str).tolist())

    def symmetry_name(self):
        """
        Laue group name.
        Spaces are removed and slashes are replaced with underscore.

        Returns
        -------
        symmetry : str
            None or Hermann-Mauguin point group symbol.

        """

        symmetry = self.params.get("Symmetry")

        name = "" if symmetry is None else "_" + symmetry.replace(" ", "")

        return name.replace("/", "_")

    def projection_name(self):
        """
        Axes projections.

        Returns
        -------
        proj : str
            Name of slices.

        """

        W = np.column_stack(self.params["Projections"])

        char_dict = {0: "0", 1: "{1}", -1: "-{1}"}

        chars = ["h", "k", "l"]

        axes = []
        for j in [0, 1, 2]:
            axis = []
            for w in W[:, j]:
                char = chars[np.argmax(W[:, j])]
                axis.append(char_dict.get(w, "{0}{1}").format(j, char))
            axes.append(axis)

        result = []
        for item0, item1 in zip(axes[0], axes[1]):
            if item0 == "0":
                result.append(item1)
            elif item1 == "0":
                result.append(item0)
            elif "-" in item1:
                result.append(item0 + item1)
            else:
                result.append(item0 + "+" + item1)

        proj = "_(" + ",".join(result) + ")" + "_[" + ",".join(axes[2]) + "]"

        return proj

    @staticmethod
    def combine_parallel(plan, files):
        instance = Normalization(plan)

        return instance.combine(files)

    def combine(self, files):
        """
        Merge data and normalization files.

        Parameters
        ----------
        files : list
            Files to be combined.

        """

        output_file = self.get_output_file()
        diag_file = self.get_diagnostic_file()

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        for ind, file in enumerate(files):
            data_file = self.get_file(file, "data")
            norm_file = self.get_file(file, "norm")

            data.load_histograms(data_file, "tmp_data")
            data.load_histograms(norm_file, "tmp_norm")

            data.combine_histograms("tmp_data", "data")
            data.combine_histograms("tmp_norm", "norm")

            bkg_data_file = self.get_file(file, "bkg_data")
            bkg_norm_file = self.get_file(file, "bkg_norm")

            if os.path.exists(bkg_data_file) and os.path.exists(bkg_norm_file):
                data.load_histograms(bkg_data_file, "tmp_bkg_data")
                data.load_histograms(bkg_norm_file, "tmp_bkg_norm")

                data.combine_histograms("tmp_bkg_data", "bkg_data")
                data.combine_histograms("tmp_bkg_norm", "bkg_norm")

                os.remove(bkg_data_file)
                os.remove(bkg_norm_file)

            os.remove(data_file)
            os.remove(norm_file)

        data_file = self.get_file(diag_file, "data")
        norm_file = self.get_file(diag_file, "norm")
        result_file = self.get_file(output_file, "")

        data.divide_histograms("result", "data", "norm")

        UB_file = file.replace(".nxs", ".mat")

        for ws in ["data", "norm", "result"]:
            data.add_UBW(ws, UB_file, self.params["Projections"])

        for ind, file in enumerate(files):
            UB_file = file.replace(".nxs", ".mat")

            os.remove(UB_file)

        data.save_histograms(data_file, "data", sample_logs=True)
        data.save_histograms(norm_file, "norm", sample_logs=True)
        data.save_histograms(result_file, "result", sample_logs=True)

        signal, error, *_ = data.extract_bin_info("result")
        UB, W, titles, axes = data.extract_axis_info("result")

        plot_path = self.get_plot_path()

        for i, vals in enumerate(axes):
            norm = np.zeros(3, dtype=int)
            norm[i] = 1

            plot_name = "slice_{}.pdf".format(titles[i].replace(" ", ""))
            pdf = Pages(os.path.join(plot_path, plot_name))

            for val in vals:
                plot = SlicePlot(UB, W)

                params = plot.calculate_transforms(signal, axes, titles, norm, val)

                coords, values, labels, T, aspect = params

                plot.make_slice(coords, values, labels, T, aspect)

                if np.isclose(np.round(val, 4) % 1, 0):
                    name = labels[2].replace(" ", "")
                    plot_name = "slice_{}.png".format(name)
                    plot.save_plot(os.path.join(plot_path, plot_name))

                pdf.add_plot()

            pdf.close()

        if mtd.doesExist("bkg_data") and mtd.doesExist("bkg_norm"):
            data_file = self.get_file(diag_file, "bkg_data")
            norm_file = self.get_file(diag_file, "bkg_norm")

            data.save_histograms(data_file, "bkg_data", sample_logs=True)
            data.save_histograms(norm_file, "bkg_norm", sample_logs=True)

            bkg_output_file = self.get_file(output_file, "bkg")

            data.divide_histograms("bkg_result", "bkg_data", "bkg_norm")
            data.save_histograms(bkg_output_file, "bkg_result")

            data.subtract_histograms("sub", "result", "bkg_result")

            sub_output_file = self.get_file(output_file, "sub_bkg")
            data.save_histograms(sub_output_file, "sub", sample_logs=True)

    def get_output_file(self):
        """
        Name of output file.

        Returns
        -------
        output_file : str
            Normalization output file.

        """

        output_file = os.path.join(
            self.plan["OutputPath"], "normalization", self.plan["OutputName"] + ".nxs"
        )

        return output_file

    def get_plot_path(self):
        """
        Plot directory.

        Returns
        -------
        plot_path : str
            Path name to save plots.

        """

        return os.path.join(self.plan["OutputPath"], "normalization", "plots")

    def get_diagnostic_file(self):
        """
        Diagnostic directory.

        Returns
        -------
        diag_path : str
            Path name to save diagnostics.

        """

        return os.path.join(
            self.plan["OutputPath"],
            "normalization/diagnostics",
            self.plan["OutputName"] + ".nxs",
        )
