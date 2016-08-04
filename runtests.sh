coverage run --source=. -m unittest discover -s nmrpy/tests/ -p "nmrpy_tests.py"
coverage report
coverage html -d nmrpy/tests/coverage_html
echo "coverage html stored in nmrpy/tests/coverage_html"
