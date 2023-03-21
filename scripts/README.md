# order of scripts to be run

**read_metadata.py**
Adds metadata to all speeches, such as speech quality, speaker name, gender, etc.

**create_timestamps.py**
Creates timestamps for all speeches with length >= 80 secs at various lengths
Saves to
	- *all_speeches_ts.parquet*
	- *all_speeches_ts_downsize.parquet*
		- a downsized version including a subset of columns
		
**split_audio_by_speeches_modified.py**
Extracts speeches given timestamps generated in previous step
		
**audio_to_embedding.py**
Extracts voiceprints of speeches extracted in previous step
Modifies *all_speeches_ts_downsize.parquet*

**bucket_data.py**
Does quality filtering of the speeches and splits them into two portions: speakers who spoke 10+ years and those who didn't.
Creates buckets for age ranges, and (tries) to fill them up with 4 speakers (2 men 2 women) per bucket.
Same as above for train+dev data, but with 20 speakers per bucket
Splits train+dev data according to an 80:20 split, but if number of speakers in an age bucket <= 4, then 100:0 split

**pipeline_demo.py**
Creates speeches pairs for bucketed speakers in 3 settings:
	- within speaker within age
	- within speaker across age
	- across speaker (no matter the age)
(TO-DO: create pairs for
	- across speaker within age
	- gender differences
	- speeches pairs for training/dev data)
	
**embs_within_age_within_speaker.py**
Does analysis based on cosine similarity score pairs from the previous file and creates graphs and tables based on that.

**tsne_embeddings.py**
Creates T-SNE graphs for bucketed speakers, and also for the subset of their speeches actually used