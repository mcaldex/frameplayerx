// Puts the contact address together in the browser: the FAQ holds it only in
// two parts, which keeps it from bots that read pages for addresses.
for (const link of document.querySelectorAll("a[data-mail]")) {
  const address = `${link.dataset.mail}@${link.dataset.at}`;
  link.href = `mailto:${address}`;
  link.textContent = address;
}
