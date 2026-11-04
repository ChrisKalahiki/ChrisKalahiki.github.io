# Security headers: GitHub Pages vs Cloudflare proxy vs Cloudflare Pages/Netlify

Research for issue #6, part of map #4. Researched 2026-11-04.

**Question.** GitHub Pages can't set response headers. Which of three routes gets
`chriskalahiki.info` HSTS, CSP, X-Frame-Options (or `frame-ancestors`), Referrer-Policy,
Permissions-Policy and X-Content-Type-Options, and an A on Mozilla Observatory and
securityheaders.com? The routes are:

1. Stay on GitHub Pages, using only a `<meta>` CSP plus hygiene.
2. Put a Cloudflare proxy in front of GitHub Pages.
3. Move hosting to Cloudflare Pages or Netlify and use a `_headers` file.

## Summary

| | 1. GH Pages + meta | 2. Cloudflare proxy -> GH Pages | 3a. Cloudflare Pages | 3b. Netlify |
|---|---|---|---|---|
| Observatory (best case) | **C (55)** | **A+** | **A+** | **A+** |
| securityheaders.com (best case) | **F** (headers only; see caveat) | A+ | A+ | A+ |
| HSTS / frame-ancestors / XFO / nosniff | Not possible | Yes | Yes | Yes |
| DNS change | Fix `www` only | Nameservers move to Cloudflare | Nameservers move to Cloudflare (apex) | Keep current DNS, repoint A/ALIAS |
| Cost | $0 | $0 | $0 | $0, with a hard credit cap |
| Main ongoing risk | None new | GitHub's Let's Encrypt renewal behind the proxy | Build drift (Ruby/gems) | Credit cap pauses the site |
| Reversibility | Trivial | Easy: grey-cloud the records or move NS back. Headers vanish | Easy-moderate | Easy-moderate |

**Leaning (not a decision):** Route 2 with SSL mode **Full**, not Full (strict). It reaches
the target grade, changes no build or hosting, and costs nothing. It's also the easiest
to undo, as long as HSTS `preload` is left off. The price is a nameserver move and a
renewal question that is only community-documented (see 2c). If you'd rather avoid that
renewal question, Cloudflare Pages is next, because it removes the GitHub cert dependency
entirely. Netlify's free tier allows about 20 production deploys a month before the site
pauses, which makes it the weakest fit.

## Current state (measured 2026-11-04)

- Mozilla Observatory scan of `chriskalahiki.info`: **F, score 10**, 5 of 12 tests failed
  (`POST https://observatory-api.mdn.mozilla.net/api/v2/scan?host=chriskalahiki.info`).
  The score is consistent with these penalties: CSP missing −25, HSTS missing −20,
  X-Frame-Options missing −20, X-Content-Type-Options missing −5, and no HTTP->HTTPS redirect −20.
- `gh api repos/ChrisKalahiki/ChrisKalahiki.github.io/pages` returns:
  - `build_type: legacy`
  - `https_enforced: false`. Plain `http://chriskalahiki.info/` returns 200 and doesn't redirect.
  - The certificate is `approved`, covers `["chriskalahiki.info"]` only, and expires `2027-02-01`.
- DNS:
  - The apex has the four GitHub A records.
  - `www` is a CNAME to `www.chriskalahiki.github.io.`, which is wrong. GitHub says it must
    point to `<user>.github.io` with no prefix
    ([GitHub: managing a custom domain](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site)).
  - Nameservers are `ns-cloud-c{1..4}.googledomains.com`.
- Response headers from GitHub Pages: `server`, `content-type`, `cache-control`,
  `access-control-allow-origin: *`, plus Fastly/Varnish headers. None of the six target headers is present.
- The page has one inline `<script>` in `_layouts/default.html` (the `toggleMenu` hamburger)
  and an inline `onclick=`. A CSP without `'unsafe-inline'` would block both. Fix them by moving
  the script to a file and attaching the listener with `addEventListener`, or by hashing the script.
  The `projects/aircanvas` page loads scripts from jsDelivr and cdnjs and has many inline
  handlers. It would need its own looser per-path policy, or a refactor.

Hygiene that applies to every route:

- Fix the `www` CNAME. Without it, the certificate won't cover `www` and GitHub can't redirect `www` to the apex.
- Turn on "Enforce HTTPS" on routes 1 and 2. It's worth +20 on Observatory's redirection test.

## How the graders score

**Mozilla HTTP Observatory.** The site starts at 100. Penalties come off first, and bonuses
are added only if the score is still at least 90. A+ means 100 or more, A means 90–99, and
C means 50–59 ([MDN: scoring methodology](https://developer.mozilla.org/en-US/observatory/docs/tests_and_scoring)).
The modifiers below come from `SCORE_TABLE` in the scanner source
([mdn/mdn-http-observatory `src/grader/charts.js`](https://github.com/mdn/mdn-http-observatory/blob/main/src/grader/charts.js)).

- CSP:
  - Missing: −25
  - `'unsafe-inline'`: −20
  - No unsafe sources: +5
  - No unsafe sources and `default-src 'none'`: +10
- HSTS:
  - Missing: −20
  - Max-age of at least 6 months: 0
  - Preloaded: +5
- X-Frame-Options:
  - Missing: −20
  - `DENY` or `SAMEORIGIN`, or `frame-ancestors` in a CSP: +5
- X-Content-Type-Options missing: −5
- Redirection missing: −20
- Referrer-Policy private: +5
- CORP `same-origin`/`same-site` and COOP `same-origin`: +10 each

Observatory does read `<meta>` tags, but only some:

- It accepts `<meta http-equiv="Content-Security-Policy">` and `<meta name="referrer">`
  ([`src/retriever/utils.js`](https://github.com/mdn/mdn-http-observatory/blob/main/src/retriever/utils.js)).
- It ignores `frame-ancestors` from a meta CSP. The scanner code comments
  "frame-ancestors can only be set via header, not via http-equiv meta tag"
  ([`src/analyzer/tests/csp.js`](https://github.com/mdn/mdn-http-observatory/blob/main/src/analyzer/tests/csp.js)).

**securityheaders.com.** It grades the HTTP response headers. Scott Helme's original scoring
post describes a header-only check, though the header set has since grown to the six named in
this issue ([scotthelme.co.uk](https://scotthelme.co.uk/scoring-transparency-on-securityheaders-io/)).
*Caveat:* I found no primary documentation of its current rubric. I also couldn't run a scan,
because automated requests get HTTP 403. My expectation that a meta-only site gets **F**,
and that an `'unsafe-inline'` CSP caps the grade below A+, is an inference, not a cited fact.

## Route 1: stay on GitHub Pages, meta CSP + hygiene

**What a `<meta>` can and can't do (per the specs)**

| Header | Via `<meta>`? | Source |
|---|---|---|
| CSP | Yes, but `frame-ancestors`, `report-uri` and `sandbox` are ignored, and report-only isn't supported | [CSP3 §3.3](https://www.w3.org/TR/CSP3/#meta-element); [MDN frame-ancestors](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy/frame-ancestors) |
| HSTS | **No**: "UAs MUST NOT heed http-equiv=\"Strict-Transport-Security\"" | [RFC 6797 §8.5](https://www.rfc-editor.org/rfc/rfc6797#section-8.5) |
| X-Frame-Options / clickjacking | **No**: not an HTML pragma, and `frame-ancestors` is ignored in meta | [HTML §4.2.5.3 pragma list](https://html.spec.whatwg.org/multipage/semantics.html#pragma-directives) |
| X-Content-Type-Options | **No** (not a pragma) | same |
| Permissions-Policy | **No** (not a pragma) | same |
| Referrer-Policy | Yes, as `<meta name="referrer" content="…">` | [HTML: meta name=referrer](https://html.spec.whatwg.org/multipage/semantics.html#meta-referrer) |

GitHub Pages has no documented way to set custom headers. The custom-domain and HTTPS docs
don't mention headers or HSTS at all
([securing with HTTPS](https://docs.github.com/en/pages/getting-started-with-github-pages/securing-your-github-pages-site-with-https)).
I spot-checked `opensource.guide` and `docs.brew.sh` on 2026-11-04, and neither sends HSTS.
`choosealicense.com` does send HSTS, but it's a GitHub-owned site. That is an observation, not a documented feature.

**Achievable grade**

- Observatory, best case:
  - Meta CSP with no unsafe sources, Enforce HTTPS on, and `<meta name="referrer">`.
  - Penalties: HSTS −20, XFO −20, X-Content-Type-Options −5. That gives **55, a C**.
  - Bonuses can't apply because the score is below 90.
  - If the CSP keeps `'unsafe-inline'`, it's 35 (D).
- securityheaders.com: likely **F** (see caveat above).

**Setup**

1. Fix the `www` CNAME to `chriskalahiki.github.io`.
2. Enable Enforce HTTPS.
3. Move the inline script and `onclick` into `assets/js/`.
4. Add `<meta http-equiv="Content-Security-Policy">` and `<meta name="referrer" content="strict-origin-when-cross-origin">` to `_layouts/default.html`.

**Maintenance:** Update the CSP whenever a page adds a third-party origin. Pages outside
the layout, like the standalone `projects/*` HTML, need their own meta tags.

**Cost:** $0. **Lock-in:** None, and fully reversible.

## Route 2: Cloudflare proxy in front of GitHub Pages

### 2a. DNS prerequisite

- Free plans need the **full** setup, meaning the nameservers move to Cloudflare. "A CNAME setup
  (partial) is only available to customers on a Business or Enterprise plan"
  ([Cloudflare: partial setup](https://developers.cloudflare.com/dns/zone-setups/partial-setup/)).
- This moves DNS hosting only. The registrar doesn't have to change.

### 2b. Setting headers on the Free plan

- **Response header Transform Rules:**
  - The Free plan allows **10 active Transform Rules**
    ([Cloudflare Transform Rules](https://developers.cloudflare.com/rules/transform/)).
  - One rule can modify **up to 30 response headers**, with static or dynamic values
    ([create rule](https://developers.cloudflare.com/rules/transform/response-header-modification/create-dashboard/)).
  - Restrictions: you can't touch `cf-*`/`x-cf-*` or `server`, and later rules overwrite earlier ones
    ([response header modification](https://developers.cloudflare.com/rules/transform/response-header-modification/)).
  - One rule, `hostname eq "chriskalahiki.info"` → Set CSP, Referrer-Policy, Permissions-Policy and
    X-Frame-Options, covers this site. A second rule can set a looser CSP for `/projects/aircanvas/*`.
- **HSTS:** Available on **all plans including Free**. Max-age runs 1–12 months, with
  includeSubDomains, preload, and a "No-Sniff Header" option that adds X-Content-Type-Options
  ([Cloudflare HSTS](https://developers.cloudflare.com/ssl/edge-certificates/additional-options/http-strict-transport-security/)).
  The same page warns that turning HTTPS off before max-age expires makes the site inaccessible.
- **"Add security headers" Managed Transform:** Adds XCTO, `x-xss-protection`, XFO `SAMEORIGIN`,
  `referrer-policy: same-origin` and `expect-ct`
  ([managed transforms reference](https://developers.cloudflare.com/rules/transform/managed-transforms/reference/)).
  It's a convenient one-click option, but it adds no CSP, HSTS or Permissions-Policy, so explicit rules are still needed.
- **Always Use HTTPS:** Available on Free
  ([docs](https://developers.cloudflare.com/ssl/edge-certificates/additional-options/always-use-https/)).

### 2c. SSL mode and GitHub's Let's Encrypt certificate behind the orange cloud

**SSL modes** ([Cloudflare SSL modes](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/))

- **Full** encrypts to the origin but doesn't validate the origin certificate.
- **Full (strict)** requires the origin certificate to be unexpired, publicly trusted (or Cloudflare
  Origin CA), and matching the hostname. Otherwise visitors get a **526**
  ([Full (strict)](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/full-strict/)).
- The default "Automatic SSL/TLS" may upgrade the zone to Full (strict) and "won't downgrade" it
  if the origin certificate later expires. Set **Custom** if you want to pin the mode (same page).

**How GitHub issues the certificate**

- GitHub runs a DNS check, then "queues a job to request a TLS certificate from Let's Encrypt"
  ([GitHub HTTPS docs](https://docs.github.com/en/pages/getting-started-with-github-pages/securing-your-github-pages-site-with-https)).
- Extra A/AAAA records, or records pointing elsewhere, "may prevent the HTTPS certificate from generating" (same page).
- Behind the orange cloud, public DNS for the apex returns Cloudflare IPs, not GitHub's.
- Let's Encrypt HTTP-01 follows up to 10 redirects and doesn't validate certificates on HTTPS
  redirects ([LE challenge types](https://letsencrypt.org/docs/challenge-types/)). The challenge
  request itself could therefore be proxied through to GitHub.
- GitHub doesn't document how its pre-check or its renewal behaves behind a proxy. Community
  reports describe issuance and renewal failing while the proxy is on, with the fix being to
  grey-cloud the records until the certificate is issued
  ([community #22790](https://github.com/orgs/community/discussions/22790),
  [community #211265](https://github.com/orgs/community/discussions/211265)).
  These are **secondary and unverified**.

**What this means in practice**

- The current certificate is valid until 2027-02-01, so Full (strict) works today.
- If GitHub's renewal fails behind the proxy, Full (strict) turns into 526 errors when the
  certificate expires. **Full** keeps serving, because it doesn't validate, at the cost of not
  authenticating the Cloudflare-to-GitHub hop.
- **Safer setup:** Get the certificate issued for apex + `www` while the records are grey-clouded,
  enable Enforce HTTPS, then switch to orange.
- After that, either:
  - pin SSL mode to **Full**, or
  - stay on **Full (strict)** and calendar a check before each roughly 90-day renewal. If renewal
    stalls, grey-cloud briefly or remove and re-add the custom domain.

**Achievable grade**

- Observatory: all six headers plus HSTS ≥6 months, with Enforce HTTPS or Always Use HTTPS, gives zero penalties.
- Bonuses (CSP +5/+10, XFO +5, Referrer +5, optional CORP/COOP +10 each) push the score past 100: **A+**.
- securityheaders.com: **A+** if the CSP has no `'unsafe-inline'`, and probably A otherwise (see caveat).

**Setup**

1. Add the zone in Cloudflare and copy over the GitHub A records and the corrected `www` CNAME.
2. Switch the nameservers away from `googledomains.com`.
3. Grey-cloud the records and confirm the GitHub certificate covers apex + `www`. Enable Enforce HTTPS.
4. Set SSL mode to Custom → Full or Full (strict).
5. Orange-cloud the records.
6. Add one or two response-header Transform Rules.
7. Enable HSTS. Start with a short max-age, raise it to 6–12 months, and leave preload off.
8. Rescan.

**Maintenance:**

- Watch the GitHub certificate renewal (see above).
- Header policy lives in the Cloudflare dashboard, not the repo, unless you manage it with Terraform or the API.
- CSP edits happen there too.

**Cost:** $0 on Cloudflare Free.

**Lock-in / reversibility:**

- Low. Grey-cloud the records, or move the nameservers back, and the site is plain GitHub Pages again. You lose only the headers.
- HSTS keeps browsers on HTTPS for its max-age. That's harmless, because GitHub Pages serves HTTPS too.
- **HSTS preload is the one-way door.** Don't enable it unless the whole domain is committed to HTTPS.

## Route 3: move hosting, `_headers` file

Both hosts need these build changes:

- Commit a `_headers` file.
- Jekyll skips underscore-prefixed files, so add `include: [_headers]` to `_config.yml`
  ([Netlify headers doc, Jekyll note](https://docs.netlify.com/manage/routing/headers/)).
- Build with Bundler using the existing `Gemfile`. `github-pages` gem v232 pins Jekyll 3.10.0
  ([pages.github.com/versions](https://pages.github.com/versions/)).
- The repo's own `.github/workflows/github-pages.yml` already builds this Gemfile on Ruby 3.4.6,
  so a modern Ruby works. Pin it with `.ruby-version`.
- `jekyll-remote-theme` fetches `pages-themes/minimal@v0.2.0` at build time, which needs network access during the build.

### 3a. Cloudflare Pages

**Headers**

- `_headers` supports up to **100 rules** with a 2,000-character line limit
  ([Pages headers](https://developers.cloudflare.com/pages/configuration/headers/)).
- It doesn't apply to responses from Pages Functions, which this site doesn't use.
- HSTS can come from `_headers` or from the zone HSTS setting.

**Build:**

- Build image v3 (the default) ships Ruby **3.4.4**. Override with `RUBY_VERSION` or `.ruby-version`
  ([build image](https://developers.cloudflare.com/pages/configuration/build-image/)).
- Build command: `bundle exec jekyll build`. Output: `_site`.

**Custom domain:**

- An **apex domain requires the zone on Cloudflare nameservers** in the same account. Subdomains only need a CNAME
  ([custom domains](https://developers.cloudflare.com/pages/configuration/custom-domains/)).
- So the DNS change is the same as route 2.
- CAA records, if added, must allow Cloudflare's CAs.

**Free tier:**

- 500 builds a month, 1 concurrent build, a 20-minute build timeout
- 20,000 files, 25 MiB per file, 100 custom domains per project
- No published bandwidth cap on the limits page

Source: [Pages limits](https://developers.cloudflare.com/pages/platform/limits/).

**Platform direction:**

- Cloudflare's migration guide says Workers has "a distinctly broader set of features" than Pages.
- `_headers` is "supported natively in Workers with static assets", so the file carries over if Pages is ever retired
  ([migrate from Pages](https://developers.cloudflare.com/workers/static-assets/migration-guides/migrate-from-pages/)).
- The guide makes no deprecation statement about Pages.

### 3b. Netlify

**Headers**

- `_headers`, or `[[headers]]` in `netlify.toml`.
- They don't apply to function or edge-function responses.
- `Server`, `Set-Cookie`, `Location` and some other headers can't be overridden.
- Source: [Netlify headers](https://docs.netlify.com/manage/routing/headers/).
- Netlify documents HSTS as something you add yourself
  ([Netlify HTTPS](https://docs.netlify.com/manage/domains/secure-domains-with-https/https-ssl/)).

**Build:** Ruby `3.x` on the Ubuntu 24.04 image. Pin it with `.ruby-version` or `RUBY_VERSION`
([build software](https://docs.netlify.com/build/configure-builds/available-software-at-build-time/)).

**Custom domain:**

- External DNS works: ALIAS/ANAME/flattened CNAME → `apex-loadbalancer.netlify.com`,
  or an A record → `75.2.60.5`, plus `www` CNAME → `<site>.netlify.app`
  ([external DNS](https://docs.netlify.com/manage/domains/configure-domains/configure-external-dns/)).
- Let's Encrypt certificates are automatic and auto-renewed.
- Nameservers don't have to move.

**Free tier:**

- "300 credits/month with credit hard limit". Credits can't be bought on the Free plan
  ([credit-based plans](https://docs.netlify.com/manage/accounts-and-billing/billing/billing-for-credit-based-plans/credit-based-pricing-plans/)).
- A production deploy costs **15 credits**, and bandwidth costs 20 credits per GB
  ([pricing](https://www.netlify.com/pricing/)).
- So about 20 deploys a month uses the whole allowance.
- When credits run out, "all projects owned by that team enter a paused state until the start
  of the next billing cycle". Visitors see "Site not available"
  ([paused projects](https://docs.netlify.com/manage/accounts-and-billing/billing/resume-paused-projects/)).

### Route 3 assessment

**Achievable grade:** **A+** on both graders, same reasoning as route 2.

**Setup**

1. Add `_headers`, the `include` entry and `.ruby-version`.
2. Connect the repo and set the build command and output directory.
3. Verify on `*.pages.dev` or `*.netlify.app`.
4. Point DNS: Cloudflare nameservers for CF Pages, or A/ALIAS for Netlify.
5. Remove the GitHub `CNAME` file, or unset the Pages custom domain, so GitHub doesn't hold the domain.
6. Disable GitHub Pages, or leave it on the `github.io` URL.

**Maintenance:**

- Headers live in the repo, which is reviewable and versioned. This is the cleanest option for CSP iteration.
- Builds can drift as the host updates its image, so pin Ruby.
- On Netlify, keep an eye on the credit meter.

**Cost:** $0 on either. Netlify's cap is a hard stop, not a bill.

**Lock-in / reversibility:**

- Moderate-low. `_headers` is host-specific in format, but trivial to port: the Cloudflare and Netlify syntax is near-identical.
- Going back to GitHub Pages means repointing DNS and losing the headers.
- CF Pages ties the apex to Cloudflare DNS.

## Open questions for the decision

- Is moving the nameservers off Google Domains/Squarespace acceptable? Routes 2 and 3a both require it, and 3b doesn't.
- Should the `projects/*` demos, like aircanvas with its CDN scripts and inline handlers, get a looser per-path CSP? Or should they be refactored to work under a strict one?
- Should HSTS preload ever be a goal? It's worth +5 on Observatory, isn't needed for A+, and is hard to reverse.
