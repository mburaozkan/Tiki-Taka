"""Schema of the ATD export: channels in the XML DataFormatSpecification vs fields in the .txt.

Prints only structure (tag names, channel ids, field counts, coverage shares) -- no values.

    python tools/atd_channels.py [tactical|broadcast]
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DEFAULT_SOURCE, require  # noqa: E402


def strip(tag):
    return tag.split("}")[-1]


def main(source):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], source=source)
    root = ET.parse(meta_path).getroot()

    # Element-tag census (names only), to spot anything beyond players/ball.
    census = collections.Counter(strip(el.tag) for el in root.iter())
    print("XML element tags:", dict(census))

    # Channel definitions (id + name attributes only).
    for el in root.iter():
        tag = strip(el.tag)
        if tag in ("Channel", "PlayerChannel", "SensorChannel") or "Channel" in tag:
            attrs = {k: v for k, v in el.attrib.items() if k.lower() in ("id", "channelid", "playerchannelid", "name")}
            if tag == "PlayerChannel":
                continue
            print("  {} {}".format(tag, attrs))
    pc = collections.Counter(el.get("channelId") for el in root.iter() if strip(el.tag) == "PlayerChannel")
    print("PlayerChannel channelIds:", dict(pc))

    # DataFormatSpecification structure: nested splits and the channel refs inside.
    for spec in root.iter():
        if strip(spec.tag) != "DataFormatSpecification":
            continue
        print("DataFormatSpecification", {k: v for k, v in spec.attrib.items() if k in ("startFrame", "endFrame")} and "(range given)")

        def walk(el, depth):
            tag = strip(el.tag)
            keep = {k: v for k, v in el.attrib.items() if k in ("separator", "playerChannelId", "channelId", "sensorChannelId")}
            if tag == "PlayerChannelRef":
                return
            kids = list(el)
            refs = collections.Counter(strip(k.tag) for k in kids if strip(k.tag).endswith("Ref"))
            ref_ids = [k.get("playerChannelId") or k.get("channelId") for k in kids if strip(k.tag).endswith("Ref")]
            print("  " * depth + "{} {} refs={} ".format(tag, keep, dict(refs)) + (str(sorted(set(i.split("_")[-1] for i in ref_ids if i))) if ref_ids else ""))
            for k in kids:
                if not strip(k.tag).endswith("Ref"):
                    walk(k, depth + 1)
        walk(spec, 1)

    # .txt structure: ':'-groups per line, ';'-items per group, ','-fields per item (counts only).
    n_lines, n_parts, items, fields = 0, collections.Counter(), collections.Counter(), collections.Counter()
    with open(raw_path) as handle:
        for line in handle:
            n_lines += 1
            parts = line.rstrip("\n").split(":")
            n_parts[len(parts)] += 1
            for g, part in enumerate(parts[1:], 1):
                sub = part.split(";")
                items[(g, len(sub))] += 1
                for s in sub[:3]:
                    fields[(g, len(s.split(",")) if s else 0)] += 1
            if n_lines >= 50000:
                break
    print("lines scanned", n_lines)
    print("':' groups per line:", dict(n_parts))
    print("(group, ';' items) counts:", dict(items.most_common(10)))
    print("(group, ',' fields per item) counts:", dict(fields.most_common(10)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE)
