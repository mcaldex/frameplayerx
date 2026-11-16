import { cp, mkdir, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildGuide } from "./guide.mjs";

const root = await realpath(fileURLToPath(new URL("../../", import.meta.url)));
const dist = join(root, "dist");
const output = join(dist, "website");

// Keep website files separate from the release archives also built in dist/.
await mkdir(dist, { recursive: true });
if (await realpath(dist) !== dist) throw new Error("dist/ must be inside this checkout");
const existingOutput = await realpath(output).catch((error) => {
  if (error.code !== "ENOENT") throw error;
});
if (existingOutput && existingOutput !== output) throw new Error("dist/website/ must be inside dist/");
await rm(output, { recursive: true, force: true });
await cp(join(root, "site"), output, { recursive: true });
await mkdir(join(output, "assets"), { recursive: true });
await cp(join(root, "assets", "logo.svg"), join(output, "assets", "logo.svg"));
// The app's own font, so the site's text looks like the headset's.
const fonts = join(root, "crates", "frameplayerx", "assets", "fonts");
await mkdir(join(output, "fonts"), { recursive: true });
for (const file of ["Inter-Regular.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf", "Inter-LICENSE.txt"]) {
  await cp(join(fonts, file), join(output, "fonts", file));
}
await buildGuide(root, output);

// Every page for search engines, except the 404 page and the page that only
// opens Frame Control.
const unlisted = new Set(["404.html", "frame-control.html"]);
const pages = (await readdir(output, { recursive: true }))
  .map((file) => file.replaceAll("\\", "/"))
  .filter((file) => file.endsWith(".html") && !unlisted.has(file))
  .map((file) => file.replace(/\.html$/, "").replace(/^index$/, ""))
  .sort();
const urls = pages.map((page) => `  <url><loc>https://frameplayerx.org.uk/${page}</loc></url>`);
await writeFile(join(output, "sitemap.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.join("\n")}
</urlset>
`);
