def test_package_import() -> None:
    import pi0_minimal

    assert pi0_minimal.__version__ == "0.1.0"
