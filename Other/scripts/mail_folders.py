"""List every IMAP folder and the newest messages in each, for when a reply
does not appear in INBOX (spam filing, or a rule moving it elsewhere)."""
from __future__ import annotations

import email
import imaplib
import sys
from email.header import decode_header, make_header
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mail  # noqa: E402


def main() -> None:
    user, password = mail.credentials()
    with imaplib.IMAP4_SSL(mail.IMAP_HOST, mail.IMAP_PORT) as imap:
        imap.login(user, password)
        _, folders = imap.list()
        names = []
        for raw in folders:
            line = raw.decode()
            names.append(line.split(' "." ')[-1].strip('"') if ' "." ' in line
                         else line.split()[-1].strip('"'))
        print("folders:", names)
        for name in names:
            try:
                status, _ = imap.select(f'"{name}"', readonly=True)
                if status != "OK":
                    continue
                _, data = imap.uid("search", None, "ALL")
                uids = data[0].split()
                print(f"\n=== {name}: {len(uids)} messages")
                for uid in uids[-4:]:
                    _, parts = imap.uid("fetch", uid, "(BODY.PEEK[HEADER])")
                    if not parts or not isinstance(parts[0], tuple):
                        continue
                    message = email.message_from_bytes(parts[0][1])
                    subject = make_header(decode_header(message["Subject"] or ""))
                    print(f"  uid {uid.decode()} | {message['Date']} | {message['From']}")
                    print(f"      {subject}")
            except Exception as exc:  # a folder we cannot select is not fatal
                print(f"{name}: {exc}")


if __name__ == "__main__":
    main()
