"""Send and check the agent mailbox described in email.txt (Purelymail).

Usage:
  python scripts/mail.py send --subject S --body-file F [--to ADDR]
  python scripts/mail.py check [--since-uid N] [--all]
"""
from __future__ import annotations

import argparse
import email
import imaplib
import smtplib
import ssl
from email.header import decode_header, make_header
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TO = "yodharinmonplub@actuarialtutor.org"  # changed from gmail 2026-09-20
SMTP_HOST, SMTP_PORT = "smtp.purelymail.com", 465
IMAP_HOST, IMAP_PORT = "imap.purelymail.com", 993


def credentials() -> tuple[str, str]:
    fields = {}
    for line in (ROOT / "email.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields["username"], fields["password"]


def send(subject: str, body: str, to: str, attachments: list[str] | None = None) -> None:
    user, password = credentials()
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    for name in attachments or []:
        path = Path(name)
        msg.add_attachment(path.read_bytes(), maintype="text", subtype="csv",
                           filename=path.name)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"sent to {to}: {subject}")


def text_of(message: email.message.Message) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
        return ""
    return message.get_payload(decode=True).decode(message.get_content_charset() or "utf-8", "replace")


def check(since_uid: int, show_all: bool) -> None:
    user, password = credentials()
    with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
        imap.login(user, password)
        imap.select("INBOX")
        criteria = "ALL" if show_all else f"UID {since_uid + 1}:*"
        _, data = imap.uid("search", None, criteria)
        uids = [int(u) for u in data[0].split() if int(u) > since_uid or show_all]
        for uid in uids:
            _, parts = imap.uid("fetch", str(uid), "(BODY.PEEK[])")
            message = email.message_from_bytes(parts[0][1])
            print(f"=== UID {uid} | {message['Date']} | from {message['From']}")
            print(f"Subject: {make_header(decode_header(message['Subject'] or ''))}")
            print(text_of(message).strip()[:4000])
        print(f"max_uid={max(uids) if uids else since_uid}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--subject", required=True)
    s.add_argument("--body-file", required=True)
    s.add_argument("--to", default=DEFAULT_TO)
    s.add_argument("--attach", nargs="*", default=[])
    c = sub.add_parser("check")
    c.add_argument("--since-uid", type=int, default=0)
    c.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.cmd == "send":
        send(args.subject, Path(args.body_file).read_text(encoding="utf-8"), args.to,
             args.attach)
    else:
        check(args.since_uid, args.all)


if __name__ == "__main__":
    main()
