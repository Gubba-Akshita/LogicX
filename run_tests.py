import importlib
import os
import sys
import traceback

if __name__ == "__main__":
    print("=" * 75)
    print("EVOFORGE-X: RUNNING COMPLETE TEST SUITE")
    print("=" * 75)

    test_dir = "tests"
    test_files = [
        f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")
    ]

    total_passed = 0
    total_failed = 0

    for tf in sorted(test_files):
        mod_name = f"tests.{tf[:-3]}"
        mod = importlib.import_module(mod_name)
        test_funcs = [
            getattr(mod, name)
            for name in dir(mod)
            if name.startswith("test_") and callable(getattr(mod, name))
        ]

        for fn in test_funcs:
            fn_name = f"{mod_name}.{fn.__name__}"
            try:
                fn()
                print(f"[PASS] {fn_name}")
                total_passed += 1
            except Exception as e:
                print(f"[FAIL] {fn_name}: {e}")
                traceback.print_exc()
                total_failed += 1

    print("=" * 75)
    print(f"TEST RESULTS SUMMARY: {total_passed} PASSED | {total_failed} FAILED")
    print("=" * 75)

    if total_failed > 0:
        sys.exit(1)
