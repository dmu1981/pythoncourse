import argparse

parser = argparse.ArgumentParser(
                    prog='clip',
                    description='Advances in AI Sample Submission - Image retrieval using Natural Language Processing (powered by OpenAI CLIP embeddings)')

parser.add_argument('action', choices=["build", "lookup"])
parser.add_argument('-p', '--path', default=".")
parser.add_argument('-t', '--text', default="A construction worker holding a stop sign")

args = parser.parse_args()

if args.action == "build":
    from build import build
    build(args.path or ".")

if args.action == "lookup":
    from lookup import lookup
    lookup(args.text)