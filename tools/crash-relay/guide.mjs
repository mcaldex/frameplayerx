// Renders docs/USER-GUIDE.md as the website's user guide: /guide lists the
// sections, and each "## " section becomes /guide/<its anchor>. Anchors are
// the ones GitHub makes, so links written for GitHub still land; links to
// other files in the repository go to GitHub.
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { join, posix } from "node:path";
import { Lexer, Marked } from "marked";

const REPO = "https://github.com/remotesojourner/frameplayerx";
const SITE = "https://frameplayerx.org.uk";
const FRAME_CONTROL = "frame-control://install?manifest=" + encodeURIComponent(`${REPO}/releases/latest/download/frameplayerx.json`);

export async function buildGuide(root, output) {
  const source = await readFile(join(root, "docs", "USER-GUIDE.md"), "utf8");
  // The site's header and footer, so the guide looks like the other pages.
  const home = await readFile(join(root, "site", "index.html"), "utf8");
  const header = home.match(/  <header class="site-header">[\s\S]*?<\/header>\n/)?.[0];
  const footer = home.match(/  <footer class="site-footer">[\s\S]*?<\/footer>\n/)?.[0];
  if (!header || !footer) throw new Error("site/index.html has no header or footer to reuse");
  const nav = header.replace('href="/guide"', 'href="/guide" aria-current="page"');

  const tokens = Lexer.lex(source);
  const seen = new Map();
  const intro = [];
  const sections = [];
  for (const token of tokens) {
    if (token.type === "heading") token.slug = slug(plain(token.tokens), seen);
    if (token.type === "heading" && token.depth === 2) {
      sections.push({ title: plain(token.tokens), slug: token.slug, tokens: [token], subs: [] });
    } else if (sections.length) {
      const section = sections.at(-1);
      section.tokens.push(token);
      if (token.type === "heading" && token.depth === 3) {
        section.subs.push({ title: plain(token.tokens), slug: token.slug });
      }
    } else {
      intro.push(token);
    }
  }
  const pageOf = new Map();
  for (const section of sections) {
    for (const token of section.tokens) if (token.type === "heading") pageOf.set(token.slug, section.slug);
  }

  const assets = new Set();
  let current = null;
  const rewrite = (target) => {
    if (target.startsWith("#")) {
      const id = target.slice(1);
      const page = pageOf.get(id);
      if (!page) {
        console.warn(`USER-GUIDE.md links to #${id}, which no heading makes`);
        return target;
      }
      if (page === current) return target;
      return page === id ? `/guide/${page}` : `/guide/${page}#${id}`;
    }
    // GitHub drops links that open apps, so the guide sends GitHub readers to
    // /frame-control; on the site the button can open Frame Control itself.
    if (target === `${SITE}/frame-control`) return FRAME_CONTROL;
    if (/^[a-z][a-z0-9+.-]*:/i.test(target)) {
      // A link to this site becomes a path on it, judged by its whole origin
      // so a look-alike host (frameplayerx.org.uk.example.com) stays as it is.
      const url = URL.canParse(target) ? new URL(target) : null;
      return url?.origin === SITE ? url.pathname + url.search + url.hash : target;
    }
    // A path in the repository, relative to docs/.
    const [path, hash = ""] = target.split(/(?=#)/);
    const file = posix.normalize(posix.join("docs", path));
    if (file.startsWith("assets/")) {
      assets.add(file);
      return `/${file}`;
    }
    return `${REPO}/blob/main/${file}${hash}`;
  };

  const marked = new Marked({
    renderer: {
      heading({ tokens: inline, depth, slug: id }) {
        // Each section is its own page, so "## " is the page's title.
        const level = Math.max(1, depth - 1);
        const text = this.parser.parseInline(inline);
        if (level === 1) return `<h1 id="${id}">${text}</h1>\n`;
        return `<h${level} id="${id}">${text} <a class="anchor" href="#${id}" aria-label="Link to this part">#</a></h${level}>\n`;
      },
      link({ href, title, tokens: inline }) {
        const text = this.parser.parseInline(inline);
        const titled = title ? ` title="${escape(title)}"` : "";
        return `<a href="${escape(rewrite(href))}"${titled}>${text}</a>`;
      },
      // Raw HTML in the guide is already escaped; only its paths change.
      html({ text }) {
        return text.replace(/\b(src|href)="([^"]*)"/g, (_, name, value) => `${name}="${rewrite(value)}"`);
      },
    },
  });
  const render = (list) => {
    const html = marked.Parser.parse(Object.assign([...list], { links: tokens.links }), marked.defaults);
    return html.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, "</table></div>");
  };

  await mkdir(join(output, "guide"), { recursive: true });
  for (const [index, section] of sections.entries()) {
    current = section.slug;
    const main = `    <div class="wrap guide">
${sidebar(sections, section)}
      <article class="prose">
${render(section.tokens)}
${pager(sections[index - 1], sections[index + 1])}
      </article>
    </div>
`;
    const html = page({
      title: `${section.title} · FramePlayerX user guide`,
      description: summary(section.tokens),
      path: `guide/${section.slug}`,
      nav,
      footer,
      main,
    });
    await writeFile(join(output, "guide", `${section.slug}.html`), html);
  }

  current = null;
  const contents = sections.map((section) => {
    const subs = section.subs.length
      ? `<ul>${section.subs.map((sub) => `<li><a href="/guide/${section.slug}#${sub.slug}">${escape(sub.title)}</a></li>`).join("")}</ul>`
      : "";
    return `          <li><a class="toc-title" href="/guide/${section.slug}">${escape(section.title)}</a>${subs}</li>`;
  });
  const main = `    <div class="wrap guide">
${sidebar(sections, null)}
      <article class="prose">
        <h1>User guide</h1>
${render(intro.filter((token) => token.type === "paragraph"))}
        <ol class="guide-contents">
${contents.join("\n")}
        </ol>
      </article>
    </div>
`;
  const description = summary(intro);
  await writeFile(join(output, "guide.html"), page({ title: "User guide · FramePlayerX", description, path: "guide", nav, footer, main }));

  for (const file of assets) {
    await mkdir(join(output, posix.dirname(file)), { recursive: true });
    await cp(join(root, file), join(output, file));
  }
}

function sidebar(sections, current) {
  const items = sections.map((section) => {
    const here = section === current;
    const subs = here && section.subs.length
      ? `<ol>${section.subs.map((sub) => `<li><a href="#${sub.slug}">${escape(sub.title)}</a></li>`).join("")}</ol>`
      : "";
    return `          <li><a href="/guide/${section.slug}"${here ? ' aria-current="page"' : ""}>${escape(section.title)}</a>${subs}</li>`;
  });
  return `      <nav class="guide-nav" aria-label="User guide">
        <a class="guide-home" href="/guide"${current ? "" : ' aria-current="page"'}>User guide</a>
        <ol>
${items.join("\n")}
        </ol>
      </nav>`;
}

function pager(previous, next) {
  const link = (section, rel) => section
    ? `<a class="${rel}" href="/guide/${section.slug}" rel="${rel}"><small>${rel === "prev" ? "Previous" : "Next"}</small>${escape(section.title)}</a>`
    : "<span></span>";
  return `        <nav class="pager" aria-label="Previous and next">
          ${link(previous, "prev")}
          ${link(next, "next")}
        </nav>`;
}

function page({ title, description, path, nav, footer, main }) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escape(title)}</title>
  <meta name="description" content="${escape(description)}">
  <meta name="theme-color" content="#0a0c16">
  <link rel="canonical" href="${SITE}/${path}">
  <link rel="icon" href="/assets/logo.svg" type="image/svg+xml">
  <meta property="og:type" content="article">
  <meta property="og:site_name" content="FramePlayerX">
  <meta property="og:title" content="${escape(title)}">
  <meta property="og:description" content="${escape(description)}">
  <meta property="og:url" content="${SITE}/${path}">
  <meta property="og:image" content="${SITE}/media/og.jpg">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="preload" href="/fonts/Inter-Regular.ttf" as="font" type="font/ttf" crossorigin>
  <link rel="preload" href="/fonts/Inter-Bold.ttf" as="font" type="font/ttf" crossorigin>
  <link rel="stylesheet" href="/site.css">
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>

${nav}
  <main id="main">
${main}  </main>

${footer}</body>
</html>
`;
}

// The text of inline tokens, without their Markdown.
function plain(tokens) {
  return tokens
    .map((token) => (token.tokens ? plain(token.tokens) : token.type === "br" ? " " : token.text ?? ""))
    .join("")
    .replace(/&(amp|lt|gt|quot|#39);/g, (_, name) => ({ amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'" })[name]);
}

// GitHub's anchor for a heading, numbered when the same text comes again.
function slug(text, seen) {
  const base = text.toLowerCase().replace(/[^\p{L}\p{M}\p{N}\p{Pc} -]/gu, "").replace(/ /g, "-");
  const count = seen.get(base) ?? 0;
  seen.set(base, count + 1);
  return count ? `${base}-${count}` : base;
}

// The first paragraph's text, or the first list item's when a section opens
// with a list, cut at a word to fit a description.
function summary(tokens, limit = 155) {
  const first = tokens.find((token) => token.type === "paragraph" || token.type === "list");
  const inline = first?.type === "list" ? first.items[0].tokens : first?.tokens ?? [];
  const text = plain(inline).replace(/\s+/g, " ").trim().replace(/:$/, ".");
  if (text.length <= limit) return text;
  return text.slice(0, text.lastIndexOf(" ", limit - 1)).replace(/[,;:]$/, "") + "…";
}

function escape(text) {
  return String(text).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}
