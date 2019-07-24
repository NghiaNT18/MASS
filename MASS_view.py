MASS-fairseq/
├── README.md
├── archi_mass_sup.png
├── ft_mass_enzh.sh
├── generate_enzh_data.sh
├── mass
│   ├── __init__.py
│   ├── masked_language_pair_dataset.py
		class MaskedLanguagePairDataset(FairseqDataset):
            def __getitem__(self, index):
            def __len__(self):
            def collater(self, samples):
            def num_tokens(self, index):
            def ordered_indices(self):
            def supports_prefetch(self):
			def prefetch(self, indices):
            def size(self, index):
            def __init__():
            def _collate(self, samples, pad_idx, eos_idx, segment_label):  # new method
            def merge(key, left_pad):  # new method
            def get_dummy_batch(  # new method
            def mask_start(self, end):  # new method
            def mask_word(self, w):  # new method
            def random_word(self, w, pred_probs):  # new method
            def mask_interval(self, l):  # new method
│   ├── noisy_language_pair_dataset.py
│   ├── xmasked_seq2seq.py
		class XMassTranslationTask(FairseqTask):
    		def add_args(parser):
    		def __init__(self, args, dicts, training):
    		def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
    		def setup_task(cls, args, **kwargs):
    		def prepare(cls, args, **kwargs):
    		def load_dataset(self, split, **kwargs):
    		    def split_exists(split, lang):
    		    def split_para_exists(split, key, lang):
    		    def indexed_dataset(path, dictionary):
    		def build_model(self, args):
    		def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
    		    def forward_backward(model, samples, logging_output_key, lang_pair, weight=1.0):
    		def valid_step(self, sample, model, criterion):
    		def inference_step(self, generator, models, sample, prefix_tokens=None):
    		def init_logging_output(self, sample):
    		def grad_denom(self, sample_sizes, criterion):
    		def aggregate_logging_outputs(self, logging_outputs, criterion):
    		    def sum_over_languages(key):
    		def max_positions(self):
    		def source_dictionary(self):
    		def target_dictionary(self):
    		def load_dictionary(cls, filename):
│   └── xtransformer.py
		class XTransformerEncoder(TransformerEncoder):
		class XTransformerDecoder(TransformerDecoder):
		class XTransformerModel(BaseFairseqModel):
classmethod
├── run_mass_enzh.sh
└── translate.sh

1 directory, 11 files
