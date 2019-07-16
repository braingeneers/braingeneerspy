test:
	# BRAINGENEERS_ARCHIVE_PATH="./tests" python3 -B -m pytest -s
	python3 -B -m pytest -s

sync:
	aws --profile prp --endpoint https://s3.nautilus.optiputer.net \
		s3 sync --delete \
		./tests/derived/test-datasets/ \
		s3://braingeneers/archive/derived/test-datasets/ \
		--acl public-read
