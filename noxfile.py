"""Build the Python package into clean environments and run the test suites.

    nox                 # rust + python tests
    nox -s tests        # Python tests only
    nox -s figures      # the phantom results and figures (examples/phantoms.py)
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
    """The phantom results and figures. The knee and benchmark pages need the
    CT and a Bonelab checkout: ``python examples/knee.py DICOM_DIR`` and
    ``python examples/benchmarks.py BONELAB_DIR``."""
    session.install("./python[docs]")
    session.chdir("examples")
    session.run("python", "phantoms.py", *session.posargs)


@nox.session
def docs(session):
    """Build the documentation (warnings are errors) and run its examples."""
    session.install("./python[docs]")
    session.run("sphinx-build", "-W", "--keep-going", "-b", "html", "docs", "docs/_build/html")
    session.run("sphinx-build", "-b", "doctest", "docs", "docs/_build/doctest")
