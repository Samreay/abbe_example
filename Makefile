install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install
	uv sync
	uv run pre-commit install
	uv run pre-commit autoupdate
	uv run pre-commit gc

precommit:
	uv run pre-commit run --all-files

run:
	uv run plot_power_spectrum.py --distance=uniform
	uv run plot_power_spectrum.py --distance=gaussian
	uv run plot_power_spectrum.py --distance=r_power_minus_2
	uv run plot_power_spectrum.py --distance=r_power_1