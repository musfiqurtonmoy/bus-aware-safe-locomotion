"""Download the public source data into data/raw/ (not redistributed with this repository).

    python scripts/download_data.py            # download what is missing, verify every file
    python scripts/download_data.py --check    # verify the files already present, download nothing

1. LG 18650HG2 Li-ion Battery Data (Kollmeyer, Naguib, Skells), Mendeley Data, version 2,
   DOI 10.17632/b5mj79w5w9.2, licence CC BY 4.0. Only the six files used by
   scripts/fit_battery.py, into data/raw/lg_hg2/, each checked against its SHA-256; then
   data/raw/lg_hg2/MANIFEST.csv is written (fit_battery.py reads it).
2. MuJoCo Menagerie, Unitree Go1 model (unitree_go1/, BSD-3-Clause, Unitree Robotics), pinned to
   commit 370c43965323c0944b433b202aac5058b5f907a9, into data/raw/menagerie/repo/. The leg model
   assets/leg.xml takes its masses and inertias from go1.xml (model_provenance.md); no script
   reads these files, they are for checking.
3. Katz (2018), "A low cost modular actuator for dynamic robots", MIT S.M. thesis: not
   redistributed and not downloaded (the repository refuses automated downloads). Where to get it
   is printed at the end. No script reads it; params.py and params_provenance.md cite its pages.

Uses only the Python standard library. Sends no personal information (generic user agent).
"""
import argparse
import csv
import hashlib
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "busaware-download/1.0"}

MENDELEY = "https://data.mendeley.com/public-files/datasets/b5mj79w5w9/files/{}/file_downloaded"
LG_HG2 = [  # path under data/raw/lg_hg2, SHA-256, Mendeley file id
    ("25degC/549_C20DisCh.csv", "52eff1a8d273f353d557af8172139c2f23709499d36130ac24ffbbb820e8f232",
     "c2472dec-04f8-4ab5-9bf8-49b616782d53"),
    ("25degC/549_HPPC.csv", "70ffb357511dd77e7c6bc14c642a095ddd51d6918b547987247100a3bd2ea29d",
     "7d99de7c-e149-48d9-a172-ef0f9fc64aed"),
    ("Readme file - Description of Experimental Tests.txt",
     "809c0045eabcf4913a380ad520e4016f52a9c4186e6d15958992df3baf0e1427", "06a06a79-d31f-4f4c-81a7-8cb38eab6769"),
    ("Technical Information and Experimental Test Results for LG 18650HG2.pdf",
     "7af5bfc6156a968316e57d8340005d7403c35d2da0eee3bbd47d8a3d41c347a5", "e0ff062b-8d2a-4ebb-81c0-1625a49b7d4c"),
    ("0degC/585_C20DisCh.csv", "1b5a67d1399f50062f067f06043e30b8919331c168b8c3516306ff64fe9b0fe3",
     "c057946c-dada-4ff1-b74f-3150eb8c47e7"),
    ("0degC/585_HPPC.csv", "bdb4aee01f909c77451602a6292932a00fa8d552409ff1233a6aff572f7a04f5",
     "35ffb527-f673-4e18-a1da-03f2264ae082"),
]

MENAGERIE_COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
MENAGERIE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{}/{}"
MENAGERIE_FILES = [  # path in the Menagerie repository, SHA-256
    ("unitree_go1/go1.xml", "555b1c3742957dee00f7d4a1f7a5788157f196ea9131bf599882274c29a50cdc"),
    ("unitree_go1/LICENSE", "ddd855c253ccc075cd16b7e1c460dfa5841d21a82647faffd96d1363161e5223"),
    ("unitree_go1/README.md", "18ef3a05f6ebfe953fb334b760c9ecb6594b608b3051e6d72adfb88ab50a7e6c"),
    ("CITATION.cff", "f5ef8df14e48f876e699a4f6f8863a72570c38bfb718ca39b220db90b2c0f8b9"),
]

KATZ_HANDLE = "https://hdl.handle.net/1721.1/118671"
KATZ_SHA256 = "be6bf4780645b138c1a32fda45fb207ce1546ded4b8f7e6d3d7f326ea47ad8d1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, expected, check_only):
    """Download url to dest unless dest already has the expected SHA-256. Returns True if verified."""
    if dest.exists() and sha256(dest) == expected:
        print(f"ok        {dest.relative_to(ROOT)}")
        return True
    if check_only:
        print(f"{'MISMATCH' if dest.exists() else 'missing':9s} {dest.relative_to(ROOT)}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"download  {dest.relative_to(ROOT)}")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        print(f"MISMATCH  {dest.relative_to(ROOT)}: SHA-256 {got}, expected {expected}")
        return False
    tmp.replace(dest)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify existing files only, download nothing")
    a = ap.parse_args()
    ok = True

    print("LG 18650HG2 data (Mendeley Data, DOI 10.17632/b5mj79w5w9.2, CC BY 4.0)")
    lg = RAW / "lg_hg2"
    rows = []
    for rel, sha, fid in LG_HG2:
        url = MENDELEY.format(fid)
        good = fetch(url, lg / rel, sha, a.check)
        ok &= good
        rows.append({"path": f"data/raw/lg_hg2/{rel}", "sha256": sha, "verified": str(good), "url": url})
    if all(r["verified"] == "True" for r in rows):
        with open(lg / "MANIFEST.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["path", "sha256", "verified", "url"], quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote     {(lg / 'MANIFEST.csv').relative_to(ROOT)}")

    print(f"\nMuJoCo Menagerie, Unitree Go1 model, commit {MENAGERIE_COMMIT} (BSD-3-Clause)")
    for rel, sha in MENAGERIE_FILES:
        ok &= fetch(MENAGERIE.format(MENAGERIE_COMMIT, rel), RAW / "menagerie" / "repo" / rel, sha, a.check)

    katz = RAW / "katz2018" / "katz2018_thesis.pdf"
    print("\nKatz (2018) thesis: not redistributed, not downloaded by this script.")
    print(f"  Get it from MIT DSpace: {KATZ_HANDLE}")
    print(f"  Optional: save it as {katz.relative_to(ROOT)} (SHA-256 {KATZ_SHA256}).")
    print("  No script reads it; params.py and params_provenance.md cite its printed pages.")
    if katz.exists():
        print(f"  present, SHA-256 {'matches' if sha256(katz) == KATZ_SHA256 else 'DIFFERS'}")

    print("\nall files verified" if ok else "\nsome files are missing or do not match (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
