Changelog
=========

v0.2.4 (2020-06-22)
-------------------

New
~~~
- Update documentation for conda install. [JM Rohwer]
- Add conda recipe for packaging. [JM Rohwer]

Fix
~~~
- Fix plot_deconv_array() arguments. [JM Rohwer]


v0.2.3 (2020-06-15)
-------------------

New
~~~
- Use ipywidgets.Output() in selection widgets for compatibility with
  ipympl 0.5.0+ [JM Rohwer]

Changes
~~~~~~~
- Update references to '%matplotlib widget' in docs and tutorial
  notebook. [JM Rohwer]

Fix
~~~
- Fix integral trace selection exceptions. [JM Rohwer]


v0.2.2 (2020-06-10)
-------------------

New
~~~
- Add test for calibrate() [JM Rohwer]
- Add tests for new Bruker array set import functionality. [JM Rohwer]
- Update documentation for phaser() and calibrate() methods. [JM Rohwer]
- Add Bruker test data for interleaved array experiment. [JM Rohwer]
- New Bruker importer that can deal with arrays of experiments, also
  with different nuclei interleaved. [JM Rohwer]
- Add .calibrate() method to FidArray for offset calibration. [JM
  Rohwer]
- Add .calibrate() method to Fid for spectrum offset calibration. [JM
  Rohwer]
- Add CHANGELOG.rst (generated with gitchangelog) [JM Rohwer]
- Add .gitchangelog.rc. [JM Rohwer]

Changes
~~~~~~~
- Update quickstart_tutorial.ipynb with modified phaser() and new
  calibrate() methods. [JM Rohwer]
- Update Phaser() to display cumulative phase angles at bottom of plot
  and update live. [JM Rohwer]
- Add link to CHANGELOG on README.md. [Johann Rohwer]
- Clean up BrukerImporter and VarianImporter to remove code duplication.
  [JM Rohwer]
- Simplify instantiation of FidArray by calling nmrpy.from_path()
  directly. [JM Rohwer]

Fix
~~~
- Fix calculation of time in plot_array() and plot_deconv_array() [JM
  Rohwer]
- Fix calculation of 'sw' in _extract_procpar_varian() [JM Rohwer]
- Make phaser widget display cumulative phase correction so that it can
  be applied in scripts. [JM Rohwer]
- Fix linebreaks in docstrings. [JM Rohwer]
- Fix Fid.plot_ppm() to display Time on x-axis if data have not been
  Fourier-transformed. [JM Rohwer]
- Fix grepping of version tags in .gitchangelog.rc. [JM Rohwer]
- Fix version number for docs, fix Phaser widget, fix data files in
  setup.py. [JM Rohwer]


v0.2.1 (2020-02-18)
-------------------

New
~~~
- Add Quickstart Tutorial jupyter notebook. [JM Rohwer]
- Update documentation, add reference to example Jupyter notebook (to be
  completed) [JM Rohwer]
- Add installation instructions to README. [JM Rohwer]

Changes
~~~~~~~
- Bump version to 0.2.1. [JM Rohwer]

Fix
~~~
- Fix bug with phaser widget that prevented saving of FidArray. [JM
  Rohwer]


v0.2 (2020-02-14)
-----------------

New
~~~
- Fix setup.py for proper packaging as wheel and tarball, add
  long_description. [JM Rohwer]
- Add instructions for tests to documentation. [JM Rohwer]
- Refactor tests so that they can be run from within the module. [JM
  Rohwer]
- Update setup.py to copy tests and test files, as well as PDF docs. [JM
  Rohwer]
- Update documentation for Version 0.2. [JM Rohwer]
- Initial work on updating docs (Quickstart tutorial) [JM Rohwer]
- Update docs (Installation) [JM Rohwer]
- Added version attribute. [JM Rohwer]
- Refactor FidArray.select_integral_traces() to work with ipympl. [JM
  Rohwer]
- Fix Fid.peakpicker(), make widget a separate class as for other
  pickers and selectors. [JM Rohwer]
- Finish Peak Trace Selector integration with ipympl. [JM Rohwer]
- Make Peak Trace Selector work with ipympl, work in progress. [JM
  Rohwer]
- Refactor baseliner widget (for FidArray) to work with ipympl. [JM
  Rohwer]
- Refactor phaser widget and baseliner widget (for Fid) to work with
  ipympl. [JM Rohwer]
- Refactor FidArray peakpicker widget to work with ipympl. [JM Rohwer]
- Move 'assign' functionality in peak picker widget to a mixin class.
  [JM Rohwer]
- Refactor Fid peak picker widget to work with ipympl. [JM Rohwer]

Changes
~~~~~~~
- Remove debugging 'print' statements from nmrpy_tests.py. [JM Rohwer]
- Update .gitignore. [JM Rohwer]
- Update copyright, authors and version increment. [JM Rohwer]
- Edit .gitignore. [JM Rohwer]
- Update author information. [JM Rohwer]
- General cleanup, replace pylab with matplotlib.pyplot as plt. [JM
  Rohwer]
- Remove unnecessary requirements. [JM Rohwer]
- Clean up FidArray.peakpicker_traces() [JM Rohwer]
- Clean up FidArray.baseliner() [JM Rohwer]
- Clean up FidArray.peakpicker() [JM Rohwer]
- Clean up Fid.baseliner() [JM Rohwer]
- Replace 'pylab' with 'plt' according to matplotlib best practice. [JM
  Rohwer]
- Make figsizes smaller so figs fit into Jupyter notebook. [JM Rohwer]
- Remove unused mlab function 'dist' [JM Rohwer]
- Edit .gitignore. [JM Rohwer]
- Update requirements to less strict versions. [JM Rohwer]
- Add .idea directory to .gitignore. [JM Rohwer]

Fix
~~~
- Fix elementwise comparison deprecation bug. [JM Rohwer]
- Fix docstrings for Sphinx. [JM Rohwer]
- Fix bug with peakpicker() and y_indices. [JM Rohwer]
- Update complex dtypes to work under win32. [JM Rohwer]
- Remove binary dist files and egg-info from version control (created
  automatically during setup) [JM Rohwer]
- Fix default offset values. [JM Rohwer]
- Fix offset. [JM Rohwer]
- Fix plt name collisions. [JM Rohwer]
- Fix instantiation of selector classes so that FidArray saves properly.
  [JM Rohwer]
- Fix bug with peaks and ranges in DataSelector() [JM Rohwer]
- Fix bug in Fid.baseline_correct() [JM Rohwer]
- Clean up passing of fid arguments to peak picker. [JM Rohwer]
- Fix requirements. [JM Rohwer]
- Fix typos in docstrings and a syntax error with 'is None' [JM Rohwer]
- Fix requirements. [JM Rohwer]
- Rename README. [JM Rohwer]
- 'is None' fixes to plotting.py. [JM Rohwer]


v0.1 (2016-09-15)
-----------------
- Initial release. [Johann Eicher]
