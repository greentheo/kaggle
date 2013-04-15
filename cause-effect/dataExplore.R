#load up and explore the data

sub = read.csv('cause-effect/CEdata_baseline_submission.csv')

train_pubinfo = read.csv('cause-effect/CEdata_text/PUBLIC/CEdata_train_publicinfo.csv')
train_pairs = read.csv('cause-effect/CEdata_text/PUBLIC/CEdata_train_pairs.csv')
train_targ = read.csv('cause-effect/CEdata_text/PUBLIC/CEdata_train_target.csv')
valid_pairs = read.csv('cause-effect/CEdata_text/PUBLIC/CEdata_valid_pairs.csv')
valid_pubinfo = read.csv('cause-effect/CEdata_text/PUBLIC/CEdata_valid_publicinfo.csv')


