"""Prepare a frozen bounded TinyStories subset and train a BPE on TRAIN stories only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders

p=argparse.ArgumentParser()
p.add_argument('--output',default='data/tinystories_lab')
p.add_argument('--train-stories',type=int,default=12000)
p.add_argument('--heldout-stories',type=int,default=1200)
p.add_argument('--vocab-size',type=int,default=4096)
a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
revision=HfApi().dataset_info('roneneldan/TinyStories').sha
splits={}
for key,n in [('train',a.train_stories),('validation',a.heldout_stories*2)]:
 ds=load_dataset('roneneldan/TinyStories',split=key,revision=revision,streaming=True)
 # Bounded shuffled sample; order and revision are recorded, not a globally uniform dataset sample.
 ds=ds.shuffle(seed=17,buffer_size=2000)
 splits[key]=[r['text'] for r in ds.take(n)]
train=splits['train']; held=splits['validation']
# Remove exact document overlaps across splits; near duplicates are not addressed here.
seen=set(hashlib.sha256(t.encode()).hexdigest() for t in train)
validation=[];test=[]
for i,text in enumerate(held):
 digest=hashlib.sha256(text.encode()).hexdigest()
 if digest in seen:continue
 seen.add(digest)
 (validation if i%2==0 else test).append(text)
tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'))
tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder=decoders.ByteLevel()
trainer=trainers.BpeTrainer(vocab_size=a.vocab_size,special_tokens=['[UNK]','[EOS]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tokenizer.train_from_iterator(train,trainer=trainer)
tokenizer.save(str(root/'tokenizer.json'))
eos=tokenizer.token_to_id('[EOS]')
encoded={}
for key,stories in [('train',train),('validation',validation),('test',test)]:
 ids=[]
 for text in stories:ids.extend(tokenizer.encode(text).ids+[eos])
 encoded[key]=np.asarray(ids,dtype=np.int32)
np.savez(root.with_suffix('.npz'),**encoded,vocab_size=tokenizer.get_vocab_size())
manifest={'dataset':'roneneldan/TinyStories','revision':revision,'shuffle_seed':17,'shuffle_buffer':2000,
          'train_stories':len(train),'validation_stories':len(validation),'test_stories':len(test),
          'token_counts':{k:len(v) for k,v in encoded.items()},'vocab_size':tokenizer.get_vocab_size(),
          'tokenizer_sha256':hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest(),
          'packed_policy':'concatenate documents with EOS; causal attention may cross EOS',
          'deduplication':'exact cross-split document hashing only; not near-duplicate decontamination'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
