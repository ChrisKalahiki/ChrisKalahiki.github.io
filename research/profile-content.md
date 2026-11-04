# Profile content: reconciled across sources

Research for #9 (map: #4). Gathered 2026-11-04.

## Sources

| Key | Source | Status |
| --- | --- | --- |
| **SITE** | This repo at `main` (`ba161e3`): `index.html`, `_pages/education.md`, `_pages/publications.md`, `_pages/work-experience.md`, `_pages/projects.md` | Read in full |
| **HAIE** | https://haielab.org/people/christopher-kalahiki/ (page footer: "last updated 2026-10-12") | Read in full |
| **GH** | `gh api users/ChrisKalahiki`, profile README (`ChrisKalahiki/ChrisKalahiki`), pinned repos (GraphQL `pinnedItems`), `gh repo list ChrisKalahiki` | Read in full |
| **GS** | https://scholar.google.com/citations?user=Y-Zcnf0AAAAJ | Read. Not blocked, but the page showed "The system can't perform the operation now. Try again later." Venues for 2 of the 3 entries were not shown. |
| **LI** | https://www.linkedin.com/in/chriskalahiki | **Blocked.** HTTP 999 (LinkedIn's bot block). None of its content appears below. Chris can supply a LinkedIn "Save to PDF" export so this source can be added. |
| **ORCID** | Public API `pub.orcid.org/v3.0/0000-0003-1513-5656/record`. The site already links this profile. | Read in full |
| **DNS** | `curl` against `chriskalahiki.info` and `www.chriskalahiki.info` | Probed |

"Stale" means the site says something that a newer source contradicts or replaces. "Missing" means another source has it and the site does not.

## Bio / research focus

| Value | Source(s) | Conflicts | Site status |
| --- | --- | --- | --- |
| Ph.D. student in Computer Science at Clemson University | SITE, HAIE, GH bio, GS ("Graduate Student, Clemson University"), ORCID | None | Current |
| Research focus: "Machine Learning and High Performance Computing" | SITE (`index.html`, education, work-experience), GS interests | HAIE lists this as Chris's *background*. Its current interests are "Trust in large language models, Human-AI teaming, Usable security, Machine learning, High-performance computing". | **Stale.** It doesn't mention trust in LLMs, human-AI teaming or usable security. |
| Member of the Human-AI Empowerment (HAIE) Lab, Clemson, "in the lab since Summer 2025". Research thrust: Inclusive AI | HAIE (news item: Aug 2025, "Three Ph.D. students join the lab") | None | **Missing** |
| Advisor: Dr. Carlos Toxtli-Hernández | HAIE | None | **Missing** |
| Dissertation topic: trust in large language models for human-AI teaming, including a study of how novice programmers team with AI assistants when "vibe coding" | HAIE | None | **Missing** |
| Graduate research assistant on the NSF SaTC 2.0 project "Tracking the Usability of Secure Authentication Systems", studying how people manage authentication in everyday life | HAIE | None | **Missing** |
| Past project areas: computer vision, explainable ML, NLP, graph learning, AI for games (the site also lists "generative models for tasks such as art and music") | SITE, HAIE | HAIE omits generative models | Current, but framed as the current focus rather than background |
| "Aloha!" greeting. Personal ML passion narrative | SITE only | None | Fine (voice, no fact to check) |
| GitHub bio: "Ph.D. student studying Computer Science at Clemson University. I am interested in exploring various applications of machine learning." | GH | Generic. Doesn't mention the HAIE focus. | n/a (GitHub's own copy is stale too) |
| Location "Central, SC". Company "Clemson University" | GH | None | Missing (probably not needed) |

## Education

| Value | Source(s) | Conflicts | Site status |
| --- | --- | --- | --- |
| Ph.D. Computer Science, Clemson University, in progress | SITE, HAIE, ORCID ("Doctorate of Philosophy", no end date) | None. No source gives a start year or an expected graduation year. | Current. Start year missing. |
| M.S. Computer Science, Louisiana Tech University, 2020 | SITE, HAIE, ORCID | None | Current |
| M.S. thesis: "Computer Vision for Inventory Management Systems", ProQuest 29257581, 2020 | SITE, GS (two entries: "Computer vision for inventory management" with 2 citations, and "Computer Vision for Inventory Management Systems", Louisiana Tech University, 2020), HAIE (described, not titled) | The title differs between the two GS entries, which are likely duplicates of each other. The site calls the M.S. thesis a "Dissertation" (`_pages/education.md`). | Current. The "Dissertation" wording is inaccurate for a master's thesis. |
| B.S. Computer Science, Louisiana Tech University, 2018 | SITE, HAIE, ORCID | None | Current |
| B.S. concentrations: Big Data & Cloud Computing, and Cyber Security | SITE only | Not confirmed elsewhere (LI blocked) | Not verifiable from the sources I could read |
| Skills and coursework lists | SITE only | Not confirmed elsewhere | Not verifiable from the sources I could read |

## Positions / work

| Value | Source(s) | Conflicts | Site status |
| --- | --- | --- | --- |
| Graduate Research Assistant, NSF SaTC 2.0 "Tracking the Usability of Secure Authentication Systems", HAIE Lab, Clemson (since Summer 2025) | HAIE | The site's only research role is a generic "Research Assistant, Clemson University School of Computing – Present" doing "research in Machine Learning and High Performance Computing" | **Stale.** The site doesn't name the lab, the advisor, the project or the start date. |
| Teaching Assistant, Clemson University, "Present" | SITE only | It sits under "Previous Experience" but is marked "Present". That is contradictory inside the site itself. | Needs confirmation. Possibly stale. |
| Graduate Research Assistant, Louisiana Tech University, 2018–2020 | SITE only | Not confirmed elsewhere | Not verifiable |
| Associate Consultant, Fenway Group, 2018–2019 | SITE (dates), HAIE (role, no dates) | None | Current |
| IT Intern, Oaklawn Racing & Gaming, 2017 (written as "2017-2017") | SITE only | Not confirmed elsewhere | Not verifiable. The date format is odd. |
| Employment: Clemson University (no role or dates) | ORCID | None | n/a |

The site's heading levels are also inconsistent ("Teaching Assistant" is `##` under "Previous Experience", and the others are `###`). That is a structural problem, not a content conflict. It is noted here for #10.

## Publications

| Value | Source(s) | Conflicts | Site status |
| --- | --- | --- | --- |
| A. Stone, C. Kalahiki, L. Li, N. Hubig, F. Iuricich, H. Dunn. "Evaluation of breast tumor morphologies from African American and Caucasian patients." *Computational and Structural Biotechnology Journal* 21 (2023), 3459–3465. doi:10.1016/j.csbj.2023.06.019 | SITE, HAIE, GS (4 citations), ORCID (the only work it lists) | None | Current |
| C. Barwulor, C. Kalahiki, E. Sidnam-Mauch, C. Toxtli-Hernández, K. Caine. "Authentication Routines Among College Students." Proceedings of the HFES Annual Meeting (ASPIRE 2026), poster | HAIE (type "poster", year 2026, with BibTeX) | Missing from GS and ORCID | **Missing.** It would replace the 2023 paper as the "Latest Publication" on `index.html`. |
| C. B. Kalahiki. "Computer Vision for Inventory Management Systems." M.S. thesis, Louisiana Tech University, 2020. ProQuest 29257581 | SITE, GS (duplicate entries, see Education) | GS has duplicates | Current |
| Scholar metrics: 6 citations, h-index 2, i10-index 0 | GS | None | Not on the site (optional) |
| Links on `publications.md` go to `https://scholar.google.com/` and `https://orcid.org/` (site homepages) | SITE | `index.html` links the correct profile URLs | **Broken links.** These should point at `?user=Y-Zcnf0AAAAJ` and `0000-0003-1513-5656`. |

## Projects

| Value | Source(s) | Conflicts | Site status |
| --- | --- | --- | --- |
| Usable Secure Authentication (HAIE lab project) | HAIE | None | **Missing** |
| IDE recommender via sentiment analysis on Stack Overflow tags | SITE (featured on `index.html`). The likely repo is `mining_stackoverflow` (public, last push 2020-09-25, no description). | The site gives no repo link. I can't confirm the repo match from metadata alone. | Needs a link. It's the oldest work but it's the "Featured Project". |
| Ruger Bot | SITE, GH `ruger-bot` (public, "Discord bot for my server", last push 2026-01-26) | None | Current |
| Image clustering with DWT on CT scans | SITE, GH `CT-Image-Analysis-using-DWT` (public, **pinned**, 5 stars, MATLAB, last push 2018-11-06) | The site lists the technology as "Python, Scikit-Learn". GitHub reports the primary language as MATLAB. | Minor conflict |
| Street sign classification CNN | SITE, GH `CNN-for-BelgiumTS-and-MNIST` (public, last push 2021-11-04) | None | Current |
| Computer Vision for Inventory Management Systems (M.S. thesis) | SITE, GS, HAIE | None | Current |
| Explainable ML research | SITE. GH `XAI-Research` is **private**. | There is no public artifact | Fine as a description. Don't link it. |
| Digimon TCG Memory Tracker (`/digimon-memory-tracker/`) | SITE, this repo | GH `digimon-memory-tracker` is **private** | Current. The hosted copy here is the only public face. |
| AirCanvas AR (`/aircanvas/`) | SITE, this repo (`projects/aircanvas/aircanvas-ar-v2.html`) | No description on the site | Current. The description is thin. |
| `lock-x` (TypeScript, public, **pinned**, no description, last push 2026-02-23) | GH | Not on the site | **Missing.** It's pinned, so Chris presumably considers it notable. |
| Recent public repos: `Brain-Dump` ("a tool to help me dump my work into my Obsidian Vault", last push 2026-11-02), `Arch-Configs`, `web-crawler`, `jax-playground`, `dcgan-with-jax`, `anime-recommendations` | GH | Not on the site | Missing. Chris should decide which belong on the site. |
| Many recent repos are private (e.g. `homeros`, `paper-bot`, `mcp-toolbench`, `context-bridge-mcp`, `Talky`, `Meta-Tracker`) | GH (`gh repo list` as owner) | n/a | Don't publish these without Chris's OK |
| The "AI for Games" and "Generative Models" blocks are commented out in `projects.md` | SITE | `index.html` and HAIE both still claim AI for games | Inconsistent inside the site |

## Contact / links

| Value | Source(s) | Conflicts | Site status |
| --- | --- | --- | --- |
| Email `ckalahi@clemson.edu` | SITE | GH profile and README use `chris.kalahiki@gmail.com`. GS verifies a clemson.edu address. | Conflict. Chris to choose. |
| LinkedIn `linkedin.com/in/chriskalahiki` | SITE, HAIE, GH README (`/in/ChrisKalahiki`) | Only the letter case differs | Current |
| GitHub `github.com/ChrisKalahiki` | SITE, HAIE | None | Current |
| Google Scholar `?user=Y-Zcnf0AAAAJ` | SITE (`index.html`), HAIE | `publications.md` links the Scholar homepage instead | Partly broken (see Publications) |
| ORCID `0000-0003-1513-5656` | SITE (`index.html`), HAIE, ORCID | `publications.md` links `orcid.org` root | Partly broken |
| Twitter/X `@ChrisKalahiki` | GH profile and README | Not on the site | Missing (Chris to decide) |
| HAIE Lab profile `haielab.org/people/christopher-kalahiki/` | HAIE | Not linked from the site | **Missing** |
| Personal site domain: `chriskalahiki.info` (CNAME in this repo). `https://chriskalahiki.info/` returns 200. | DNS, `CNAME` | HAIE, the GH profile `blog` field and ORCID all link `www.chriskalahiki.info`. Over **https** that hostname fails TLS (cert is `*.github.io`, "no alternative certificate subject name matches"). Over http it 301s to the apex. | **Broken inbound links.** Fix the GitHub Pages / DNS setup for `www`, or change those three profiles to the apex domain. |

## Biggest stale items and conflicts

1. The research focus and current role are out of date. The site says "ML and HPC" with a generic RA role. HAIE says HAIE Lab (since Summer 2025), advisor Carlos Toxtli-Hernández, NSF SaTC usable-authentication GRA, and a dissertation on trust in LLMs for human-AI teaming.
2. The HFES ASPIRE 2026 poster "Authentication Routines Among College Students" is missing from the site, GS and ORCID.
3. `https://www.chriskalahiki.info` (linked from HAIE, GitHub and ORCID) fails TLS.
4. `publications.md` links the Scholar and ORCID homepages, not Chris's profiles.
5. Email conflict: clemson.edu on the site vs gmail on GitHub.
6. The Teaching Assistant role is listed as "Present" under "Previous Experience".
7. The pinned repo `lock-x` is not on the site, and the featured IDE-recommender project has no link.

## Open questions for Chris

1. Can you provide the LinkedIn "Save to PDF" export? Every LinkedIn row above is unverified until then.
2. Which email should the site show: `ckalahi@clemson.edu` or `chris.kalahiki@gmail.com`?
3. Are you still a Teaching Assistant at Clemson? If you were, what were the dates?
4. What years did the Ph.D. start, and what is the expected graduation year? Should the site show them?
5. Should the HPC/ML framing stay as "background" (HAIE's framing), or stay in the headline focus next to trust in LLMs and usable security?
6. What is `lock-x`? Should it, `Brain-Dump`, or any other recent repo be featured? Is the IDE recommender the `mining_stackoverflow` repo?
7. Are the B.S. concentrations, the 2018–2020 Louisiana Tech GRA, the Oaklawn internship, and the skills and coursework lists still accurate and worth keeping?
8. Is the vibe-coding study published or under review yet? Is there anything to list besides the HFES poster?
9. Should the site link the HAIE profile and Twitter/X?
10. Do you want to merge the duplicate thesis entries on Google Scholar and add the HFES poster there and on ORCID? These are off-site fixes.
11. Should `www.chriskalahiki.info` be fixed (GitHub Pages custom-domain / DNS), or should the HAIE, GitHub and ORCID links be changed to the apex domain?
