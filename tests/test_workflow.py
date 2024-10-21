import os
import subprocess
import sys

import pytest

import garnet.workflow
from garnet.workflow import main as main


def test_template(tmpdir):
    script = os.path.abspath("./src/garnet/workflow.py")
    command = ["python", script]

    for inst in garnet.workflow.inst_dict.keys():
        fname = inst + ".yaml"
        filepath = os.path.join(tmpdir, fname)
        subprocess.run(command + [filepath, "temp", inst], check=False)
        assert os.path.exists(filepath)


def test_main_few_args():
    main_args = ["int", 4]
    sys.argv.append(main_args)
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == -1
