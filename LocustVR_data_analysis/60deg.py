import matplotlib.pyplot as plt
import numpy as np

# a = np.linspace(0, 50, 100)
# x = np.cos(np.radians(60)) * a
# y = np.sin(np.radians(60)) * a
# plt.plot(x, y)
# plt.show()

def access_utilities(utilities_name='utilities', resolve_parent_directories=0):
    from pathlib import Path
    import sys
    cwd = Path.cwd()
    print(cwd)
    parent_dir = cwd.resolve().parents[resolve_parent_directories]
    print(parent_dir)
    path_utilities = Path(parent_dir) / utilities_name
    sys.path.insert(0, str(path_utilities))
    print(path_utilities)

access_utilities(utilities_name='utilities')