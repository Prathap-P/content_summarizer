import re
import json
import html
import requests
from xml.etree import ElementTree as ET

VIDEO_ID = "cdLvFIRhlJ0"
OUTPUT_FILE = "subs.srt"


def get_caption_url(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    html_text = requests.get(url).text

    match = re.search(r'"captionTracks":(\[.*?\])', html_text)
    if not match:
        raise Exception("No captions found")

    tracks = json.loads(match.group(1))

    # pick English auto captions
    for t in tracks:
        if "en" in t.get("languageCode", ""):
            return html.unescape(t["baseUrl"])

    # fallback to first track
    return html.unescape(tracks[0]["baseUrl"])


def xml_to_srt(xml_text):
    root = ET.fromstring(xml_text)
    srt = []
    index = 1

    for child in root:
        start = float(child.attrib["start"])
        dur = float(child.attrib.get("dur", 0))

        end = start + dur

        text = "".join(child.itertext())
        text = html.unescape(text).replace("\n", " ")

        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        srt.append(f"{index}")
        srt.append(f"{fmt(start)} --> {fmt(end)}")
        srt.append(text)
        srt.append("")

        index += 1

    return "\n".join(srt)


def main():
    print("Fetching captions URL...")
    caption_url = get_caption_url(VIDEO_ID)

    print("Downloading captions...")
    xml_text = requests.get(caption_url).text

    print("Converting to SRT...")
    srt = xml_to_srt(xml_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(srt)

    print(f"Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()