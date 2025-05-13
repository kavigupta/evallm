import os
import shutil
import subprocess


top_level_to_copy = [
    "cache",
    "evallm",
    "notebooks",
    "output",
    "scripts",
    ".gitignore",
    ".pylintrc",
    "README.md",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
]

top_level_to_not_copy = [
    # dev stuff
    ".github",
    "dist",
    ".git",
    "env",
    "Untitled.ipynb",
    "tests",
    # temp files
    "evallm.egg-info",
    ".ipynb_checkpoints",
    "neurosym.egg-info",
    ".pytest_cache",
    # pictures that aren't outputs
    "infographics",
    "pallette.svg",
    # export process itself
    "exported_code",
    "exported_code.zip",
]

specific_exclusions = ["scripts/export.py", "output/verifai-poster.*"]

output_path = "exported_code"


def copy_directory(directory):
    if os.path.isfile(directory):
        shutil.copy2(directory, os.path.join(output_path, directory))
        return
    shutil.copytree(
        directory,
        os.path.join(output_path, directory),
    )


def delete_all_pycaches():
    for root, dirs, files in os.walk(output_path):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        for f in files:
            if f.endswith(".pyc"):
                os.remove(os.path.join(root, f))


def main():
    shutil.rmtree(output_path, ignore_errors=True)
    shutil.rmtree(output_path + ".zip", ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    for x in os.listdir("."):
        if x in top_level_to_copy:
            copy_directory(x)
        else:
            assert (
                x in top_level_to_not_copy
            ), f"Unexpected file {x} in top level directory. Please add it to the list of files to not copy."

    for x in specific_exclusions:
        cmd = f"rm -r {output_path}/{x}"
        subprocess.check_call(cmd, shell="/usr/bin/fish")

    delete_all_pycaches()

    # ensure the string 'kavi' appears nowhere in the export
    for root, dirs, files in os.walk(output_path):
        assert (
            "kavi" not in root
        ), f"Found 'kavi' in {root}. Please remove it from the export."
        for f in dirs + files:
            assert (
                "kavi" not in f
            ), f"Found 'kavi' in {os.path.join(root, f)}. Please remove it from the export."
        for file in files:
            with open(os.path.join(root, file), "rb") as f:
                contents = f.read().lower()
                assert (
                    b"kavi" not in contents
                ), f"Found 'kavi' in {os.path.join(root, file)}"

    # zip the directory
    shutil.make_archive(output_path, "zip", output_path)
    # remove the directory
    shutil.rmtree(output_path, ignore_errors=True)


if __name__ == "__main__":
    main()
