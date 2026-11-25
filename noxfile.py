"""Build the Python package into clean environments and run the test suites.

    nox                 # rust + python tests
    nox -s tests        # Python tests only
    nox -s figures      # regenerate docs/images
    nox -s docs         # build the user documentation into docs/_build/html
"""

import nox

nox.options.sessions = ["rust", "tests"]


@nox.session(python=False)
def rust(session):
    session.run("cargo", "test", "--workspace", external=True)
    session.run("cargo", "clippy", "--all-targets", "--", "-D", "warnings", external=True)


@nox.session
def tests(session):
    session.install("./python[test]")
    session.run("pytest", "python/tests", *session.posargs)


@nox.session
def figures(session):
    """Regenerate docs/images (and the benchmark JSON first with `cargo run --release --example benchmark`)."""
    session.install("./python[docs]")
    session.run("python", "docs/scripts/make_figures.py", *session.posargs)


@nox.session
def docs(session):
    """Build the documentation (warnings are errors) and run its examples."""
    session.install("./python[docs]")
    session.run("sphinx-build", "-W", "--keep-going", "-b", "html", "docs", "docs/_build/html")
    session.run("sphinx-build", "-b", "doctest", "docs", "docs/_build/doctest")
