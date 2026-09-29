# Pinned local MMseqs2 + UniRef90 route

Status: the preparation and toolcheck contract below is reproducible, and the
campaign adapters and profiles do not yet execute it. No full UniRef90 download
or full database build has been run against this contract.

Facts and links in this file were checked against the official upstream sources
on 2026-08-29.

## Frozen identity

Use this pair for the first qualification run:

| Component | Frozen identity | Upstream byte identity |
|---|---|---|
| MMseqs2 | tag `18-8cc5c`; `mmseqs version` prints commit `8cc5ce367b5638c4306c2d7cfc652dd099a4643f` | Linux AVX2 archive `mmseqs-linux-avx2.tar.gz`, 17,375,777 bytes, SHA-256 `bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce` |
| UniRef90 | UniProt release `2026_02`, dated 2026-06-10, 121,389,642 clusters | `uniref90.fasta.gz`, 32,059,052,376 bytes, upstream MD5 `abdd341aeafa7fa060c8d6639d594990`; compute and record SHA-256 after download |

The MMseqs2 values come from the official [release page](https://github.com/soedinglab/MMseqs2/releases/tag/18-8cc5c) and its GitHub release-asset metadata. The static AVX2 archive is the route pinned here; do not use `mmseqs.com/latest`, an unqualified package-manager build, or the mutable `ghcr.io/soedinglab/mmseqs2` tag.

The UniRef90 release number comes from UniProt's official [root release metalink](https://ftp.uniprot.org/pub/databases/uniprot/current_release/RELEASE.metalink) and [release notes](https://ftp.uniprot.org/pub/databases/uniprot/current_release/relnotes.txt). The FASTA size and upstream MD5 come from the official [UniRef90 metalink](https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref90/RELEASE.metalink). UniProt's [UniRef90 README](https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref90/README) describes the FASTA and says the database is CC BY 4.0.

The small `uniref90.release_note` distributed with `2026_02` has an empty release/date field. Do not use that file alone as the release identity. Preserve it for evidence, but gate the fetch on the root release notes and both metalinks.

## Trust boundary and preparation

The UniProt `current_release` URL is mutable. On 2026-08-29, `2026_02` was not yet present under `previous_releases`, so there was no release-numbered upstream URL for the 30 GB FASTA. Make the mutable URL safe as follows:

1. In one preparation transaction, fetch the four small metadata files first.
2. Require their byte sizes and SHA-256 values to match this table before downloading the FASTA.
3. Download the FASTA, require the metalink byte size and MD5, compute SHA-256, and copy the verified bytes to immutable/content-addressed storage.
4. If any metadata has changed before the initial fetch completes, stop. Do not claim newly fetched bytes are `2026_02`; select and review the new release instead.

| Retained metadata name | Bytes | SHA-256 |
|---|---:|---|
| `root.RELEASE.metalink` | 2,811 | `d82ecb077c2380265092c0038f7ea383a4647f4e488cb976270eb91c3c799990` |
| `relnotes.txt` | 1,146 | `ae71660a09dc2100a1e0d8f1ca61fdf73805a3d23a1fb295490deef51d8909bb` |
| `uniref90.RELEASE.metalink` | 4,391 | `e3cb6d885a451b340e472056b285ffc612e432819105ba006ad9451b97876336` |
| `uniref90.release_note` | 303 | `2f16174f2f95fdb70d0e44b26a7ccb76579407bd0161d32a3c5f64f9b24f7512` |

The preparation worker needs outbound HTTPS only to:

- `github.com` and GitHub's release-asset redirect host for the MMseqs2 archive;
- `ftp.uniprot.org` for UniProt metadata and `uniref90.fasta.gz`.

The known transfer payload is 32,076,436,804 bytes (29.9 GiB) plus HTTP/TLS overhead: 32,059,052,376 bytes of UniRef90, 17,375,777 bytes of MMseqs2, and 8,651 bytes of UniProt metadata. Retries must use byte ranges and still finish with the complete-file digest checks.

On an x86-64 Linux build worker with AVX2, verify and unpack the binary using the fixed URL, not a latest redirect:

```bash
curl --fail --location --retry 3 \
  --output mmseqs-linux-avx2.tar.gz \
  https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-avx2.tar.gz
printf '%s  %s\n' \
  bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce \
  mmseqs-linux-avx2.tar.gz | sha256sum --check --strict
tar --extract --gzip --file mmseqs-linux-avx2.tar.gz
./mmseqs/bin/mmseqs version
```

The last command must print the full commit above. Also require `grep -q avx2 /proc/cpuinfo` on the target worker. A different architecture needs its own release-asset name, official digest, manifest, and qualification; it is not the bundle specified here.

## Database build

Keep the verified compressed FASTA, the extracted executable, and the derived database in one immutable bundle. A reference CPU build uses MMseqs2's copy mode and disables input shuffling so the transformation is explicit:

```bash
mmseqs createdb sources/uniref90.fasta.gz db/uniref90 \
  --dbtype 1 \
  --createdb-mode 0 \
  --shuffle 0 \
  --write-lookup 1 \
  --compressed 1 \
  --threads 16
```

Record the literal argv, the MMseqs2 commit, thread count, input SHA-256, every output filename/size/SHA-256, total derived-database bytes, and a tree digest. Do not assume that rebuilding on another MMseqs2 release yields the same bytes.

Count amino-acid residues while streaming the verified FASTA and record that number as `R`. This is a required sizing fact, not an estimate. The official MMseqs2 [user guide](https://mmseqs.com/latest/userguide.pdf) models the k-mer index memory as approximately `7 * R + 8 * a^k` bytes and says an on-disk precomputed index avoids recomputing that index but should be kept on local disk rather than slow NFS. It also states that search can split a target database when memory is limited.

A permanent index is optional for correctness and recommended for a repeatedly reused campaign database. Build it only after the created database is hashed and a capacity check based on measured `R` passes:

```bash
mmseqs createindex db/uniref90 build-index-tmp \
  -s 7.5 \
  --split-memory-limit 96G \
  --threads 16 \
  --remove-tmp-files 1
```

The `96G` example assumes a separately qualified worker with materially more than 96 GB RAM. It is not compatible with the current 16 GB MSA-builder profile. If the index is built with a different memory limit or any other different argv, record that exact argv. Hash the resulting `.idx*` files into the same tree manifest. A no-index bundle records `createindex: null`; job searches then need local scratch and rebuild their prefilter data.

Do not state a fixed final database size before the build. The only source size known without downloading is 32,059,052,376 bytes, and the index can be many times larger. Admission must use:

- exact free bytes on the intended local volume;
- the measured created-database bytes;
- measured `R` and the upstream index formula for the pre-build estimate;
- exact index and peak scratch bytes captured by the qualification build;
- enough headroom to keep the old bundle during a versioned update.

For search-result scratch, the user guide gives a worst-case prefilter estimate of `21 * Q * L` bytes, where `Q` is query count and `L` is `--max-seqs`; alignments with `-a` need roughly 80 bytes per accepted record. Use those formulas for each batch and record peak disk in the canary.

## Bundle manifest and toolcheck

The bundle manifest has this identity-bearing shape. Fields marked `POPULATE` are filled only from the verified build output:

```json
{
  "schema_version": 1,
  "bundle_id": "mmseqs2-18-8cc5c-uniref90-2026_02",
  "job_time_network": "none",
  "mmseqs": {
    "tag": "18-8cc5c",
    "commit": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
    "artifact": {
      "name": "mmseqs-linux-avx2.tar.gz",
      "bytes": 17375777,
      "sha256": "bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce"
    },
    "executable": "bin/mmseqs",
    "executable_sha256": "POPULATE"
  },
  "uniref90": {
    "release": "2026_02",
    "release_date": "2026-06-10",
    "cluster_count": 121389642,
    "fasta": {
      "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref90/uniref90.fasta.gz",
      "bytes": 32059052376,
      "upstream_md5": "abdd341aeafa7fa060c8d6639d594990",
      "sha256": "POPULATE",
      "retained_path": "sources/uniref90.fasta.gz"
    },
    "metadata": {
      "root.RELEASE.metalink": {"bytes": 2811, "sha256": "d82ecb077c2380265092c0038f7ea383a4647f4e488cb976270eb91c3c799990", "retained_path": "sources/root.RELEASE.metalink"},
      "relnotes.txt": {"bytes": 1146, "sha256": "ae71660a09dc2100a1e0d8f1ca61fdf73805a3d23a1fb295490deef51d8909bb", "retained_path": "sources/relnotes.txt"},
      "uniref90.RELEASE.metalink": {"bytes": 4391, "sha256": "e3cb6d885a451b340e472056b285ffc612e432819105ba006ad9451b97876336", "retained_path": "sources/uniref90.RELEASE.metalink"},
      "uniref90.release_note": {"bytes": 303, "sha256": "2f16174f2f95fdb70d0e44b26a7ccb76579407bd0161d32a3c5f64f9b24f7512", "retained_path": "sources/uniref90.release_note"}
    }
  },
  "database": {
    "prefix": "db/uniref90",
    "commands": {"createdb": ["mmseqs", "createdb", "..."], "createindex": null},
    "residue_count": "POPULATE",
    "bytes": "POPULATE",
    "files": [{"path": "db/uniref90", "bytes": "POPULATE", "sha256": "POPULATE"}],
    "tree_sha256": "POPULATE"
  }
}
```

The example abbreviates `commands` and `files`; the actual manifest must list the complete literal argv and every file beginning with the database prefix, including the core body/header/index/lookup/source files and any `.idx*` files.

Run the new checker after materialization with network disabled:

```bash
python scripts/validation/check_mmseqs2_uniref90.py \
  --manifest /bundle/manifest.json \
  --bundle-root /bundle \
  --deep
```

The deep check hashes the retained FASTA and every derived database file, verifies both UniProt's MD5 and the locally stronger SHA-256, runs `mmseqs version`, and runs `mmseqs dbtype` against the database. Save the printed manifest SHA-256 outside the bundle. Campaign jobs run the same checker without `--deep`, with `--expected-manifest-sha256 <saved-digest>`, on a read-only immutable mount. The default check still hashes the small executable, checks all file sizes and the declared file set, and makes zero network calls.

The job sandbox must set the provider network policy to no network, not merely omit an allowlist. Mount the database and executable read-only; give only the query/result scratch directory write access. Invoke `search`, `result2msa`, `unpackdb`, `easy-search`, or `convertalis`, never the networked `mmseqs databases` downloader. No job-time DNS or HTTP host is required.

## Local target-MSA command contract

This is an unpaired, target-chain-only MSA against the pinned UniRef90 database. It is not numerically equivalent to the current public-server route, whose `mode=env` combines other databases. Record route and database identity so results from the two routes are never pooled silently.

For one or more named target queries:

```bash
mmseqs createdb target-queries.fasta work/query \
  --dbtype 1 --createdb-mode 0 --shuffle 0 --write-lookup 1 --threads 1
mmseqs search work/query /bundle/db/uniref90 work/result work/search-tmp \
  -s 7.5 -e 0.001 --max-seqs 10000 -a \
  --split-memory-limit 48G --threads 16 --remove-tmp-files 1
mmseqs result2msa work/query /bundle/db/uniref90 work/result work/msa \
  --msa-format-mode 6 --filter-msa 0 --skip-query 0 --threads 16
mmseqs unpackdb work/msa work/a3m \
  --unpack-name-mode 1 --unpack-suffix .a3m
```

The `48G` split limit is an example for a host with more RAM; it must be replaced by the qualified value and the profile resource request must exceed it. Freeze and record `-s`, E-value, `--max-seqs`, split limit, threads, and every default made explicit above. Require exactly one A3M per target ID, the query as the first sequence, a query sequence matching the input after A3M insertion removal, and at least depth one. Pass the result through the adapter's existing `clean_a3m_rows`, then record:

- MMseqs2 commit and bundle-manifest SHA-256;
- UniRef90 release, raw FASTA SHA-256, and database-tree SHA-256;
- literal search/result2msa argv;
- query SHA-256, A3M SHA-256, depth, and route `local`.

The first qualification must use an exact self-query extracted from the retained UniRef90 FASTA. It must return a full-length `fident=1`, `qcov=1` hit and an A3M whose first row is the query. Also run the campaign's fixed target-MSA anchors and compare depth/content stability with a repeat run.

## UniRef90 novelty command contract

The published campaign rule has two sequence rejection branches:

1. `fident > 0.60` and candidate/query coverage `qcov > 0.50`; or
2. gapped local `fident >= 0.30` and `alnlen >= 40`.

Search broadly enough to evaluate both branches, and apply the strict/inclusive boundaries after parsing rather than approximating them with MMseqs2 threshold flags:

```bash
mmseqs easy-search candidate-batch.fasta /bundle/db/uniref90 novelty.tsv novelty-tmp \
  -s 7.5 -e 1000000 --min-seq-id 0.30 -c 0 --cov-mode 2 \
  --max-seqs 100000 --max-rejected 2147483647 -a \
  --split-memory-limit 48G --threads 16 --remove-tmp-files 1 \
  --format-mode 4 \
  --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits
```

MMseqs2's pinned [command definitions](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/MMseqsBase.cpp) define `easy-search`, `createdb`, `convertalis`, and the database arguments; the pinned [parameter definitions](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/Parameters.cpp) define `fident`, coverage modes, `--msa-format-mode 6`, and unpacking options. The user guide states that `-a` computes the sequence-identity/backtrace information and documents the sensitivity and result-size tradeoffs of `--max-seqs`.

The `100000` cap is the operational v1 choice, not a proof of exhaustive search. Qualify its false-negative behavior on the fixed novelty anchor set before enabling the gate; if the anchor study changes sensitivity or the cap, that is a new tool revision. The parser writes a pass/fail scalar for the existing filter contract plus a detailed per-candidate record containing the best rejecting hit, both criterion booleans, `fident`, `alnlen`, `qcov`, target ID, E-value, and all bundle/argv identities.

Run the same predicate separately against the versioned known-binder corpus and OR the verdicts. UniRef90 does not satisfy the still-open known-binder-corpus requirement.

## Update policy

- Never follow `current_release` automatically. `2026_02` stays frozen for every run that names this tool revision.
- Review a new UniProt release only as a versioned migration. UniProt says releases are roughly every eight weeks, but schedule drift is not a reason to update mid-campaign.
- Build a new side-by-side bundle, deep-check it, run the exact self-query and fixed MSA/novelty anchors, and report every changed verdict and material MSA-depth change.
- Require scientific approval before a profile points to the new manifest digest. Never overwrite or delete the prior bundle while runs still cite it.
- A change to MMseqs2, UniRef90, search flags, index construction, or novelty predicate is a new tool/reference revision even if the human-facing route name remains `local`.

## What this route needs from you

The shipped adapter refuses the local route, so no profile executes the contract above yet. `claude_binder/adapters/target_msa_builder.py` raises `the local route is unavailable because this adapter has no MMseqs2 database path or execution implementation`, and names `--route public-server` with `--allow-public-msa`, or `--route precomputed`, instead. The shipped `filter-novelty` stage reads `sequence_novelty` from a
target-chain sliding-window screen, which is an ungapped window surrogate rather than a UniRef90
search. MMseqs2 and UniRef90 stay `operator_configured` rather than deployed until both change.

Two pieces of preparation are yours in any case.

- **Size the worker before the build.** The `48G` split-memory limit in the commands above cannot
  run under the 16 GB MSA-builder profile. Raise that resource request, or qualify a smaller split
  limit and a longer runtime against the full database.
- **Measure your own capacity.** Record build and search wall time, RAM, persistent bytes, peak
  scratch, and CPU type from a full-database run before calling this route PASS.
