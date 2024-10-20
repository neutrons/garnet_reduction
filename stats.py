import pstats
p = pstats.Stats('tests/profile.stats')
p.strip_dirs()
p.sort_stats('tottime')
p.print_stats(100)