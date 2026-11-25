"""Run in a compute allocation; use a local package for offline inference."""
import argparse
from filter_verse_embeddings import FilterVerseEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='University-of-Helsinki/filter-verse-embeddings')
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
model = FilterVerseEncoder.from_pretrained(args.model, device=args.device)
verses = ['Metsä tuulessa laulaa.', 'Joki kantaa illan valon.']
print('Isolated:', model.encode_verses(verses))
print('Contextual:', model.encode_poem(verses))
