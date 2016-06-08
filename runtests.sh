coverage run --source=. -m unittest discover -s tests/ -p "nmrpy_tests.py"
coverage report
coverage html -d tests/coverage_html
echo "coverage html stored in tests/coverage_html"
