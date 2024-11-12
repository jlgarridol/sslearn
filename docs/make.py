#!/usr/bin/env python3
from pathlib import Path
import shutil
import textwrap
import base64

from jinja2 import Environment
from jinja2 import FileSystemLoader
from markupsafe import Markup
import pygments.formatters.html
import pygments.lexers.python

import pdoc.render

here = Path(__file__).parent.parent

# Ignore set_score_request in the docs


if __name__ == "__main__":

    favicon = (here / "docs" / "sslearn_mini.webp").read_bytes()
    favicon = base64.b64encode(favicon).decode("utf8")
    logo = (here / "docs" / "sslearn.webp").read_bytes()
    logo = base64.b64encode(logo).decode("utf8")
    print("#"*100)
    print(here / "src" / "sslearn")
    print("#"*100)
    # Render main docs
    pdoc.render.configure(

        favicon="data:image/webp;base64," + favicon,
        logo="data:image/webp;base64," + logo,
        logo_link="/sslearn",
        footer_text=f"pdoc {pdoc.__version__}",
        search=True,
        math=True,
        include_undocumented=False,
        docformat="numpy",
    )
    pdoc.pdoc(
        here / "src" / "sslearn",
        output_directory=here / "docs",
    )

    
    with (here / "docs" / "sitemap.xml").open("w", newline="\n") as f:
        f.write(
            textwrap.dedent(
                """
        <?xml version="1.0" encoding="utf-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">
        """
            ).strip()
        )
        for file in here.glob("**/*.html"):
            if file.name.startswith("_"):
                continue
            filename = str(file.relative_to(here).as_posix()).replace("index.html", "")
            f.write(f"""\n<url><loc>https://pdoc.dev/{filename}</loc></url>""")
        f.write("""\n</urlset>""")
