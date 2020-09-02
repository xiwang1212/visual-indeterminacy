"""
 Toward Quantifying Ambiguities in Artistic Images - Python analysis implementation
 Copyright (c) 2020 Xi Wang <xi.wang@tu-berlin.de>

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import altair as alt
from autocorrect import Speller
from collections import Counter
from cube.api import Cube
import itertools
import math
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import streamlit as st
import time


####################################################################################
deny_list = ['image', 'photo', 'art', 'artwork', 'something', 'thing', 'stuff', 'painting', 'abstract', 'shape', 'kind',
			 'picture']
times = [500, 3000]

study_names = ['Indeterminate', 'Recognizable', 'Abstract', 'Dichotomous', 'AbstractFlat']

MAX_DISTANCE = 22
HIER_THRESH = 2
FREQUENCY_THRESH = 1
leave_out_FILTERED = True

stopword_file_path = "./SmartStoplist.txt"
stopword_list = []

datapath = '../study_data/%s.pickle'
histogram_picklefile_gf = '../res/histograms_grouped_%s_%d.pickle'

spell = Speller(lang='en')
cube = None


####################################################################################
@st.cache
def initNLTK():
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('brown')
	nltk.download('wordnet')

	return nltk.PorterStemmer()


####################################################################################
# File IO
def get_raw_data():

	f = open(datapath % studyname, 'rb')
	DATA = pickle.load(f)
	D_coords_fixated = DATA['D_coords_fixated']
	return D_coords_fixated


def loadStopwords(stop_word_file):

	stop_words = []
	for line in open(stop_word_file):
		if line.strip()[0:1] != "#":
			for word in line.split():  # in case more than one per line
				stop_words.append(word)
	return stop_words

####################################################################################
# utils
def merge_histograms(picklefile):

	coords_fixated = {}
	histogram = {}
	entropy = {}
	entropy_frames = []

	for s in study_names:
		with open(picklefile % (s, FREQUENCY_THRESH), 'rb') as file:
			p = pickle.load(file)
		D_coords_fixated = p[0]
		D_histogram = p[1]
		D_entropy = p[2]
		D_entropy_df = p[3]

		coords_fixated = {**coords_fixated, **D_coords_fixated}
		histogram = {**histogram, **D_histogram}
		entropy = {**entropy, **D_entropy}
		entropy_frames.append(D_entropy_df)

	entropy_df = pd.concat(entropy_frames)
	return coords_fixated, histogram, entropy, entropy_df


@st.cache
def compute_entropy(hist):
	ent = 0
	for i in range(len(hist)):
		if hist[i] != 0:
			ent -= hist[i] * math.log2(abs(hist[i]))
	return ent


def get_elem_ranks(curlist):
	N = len(curlist)
	output = [0] * N
	for i, x in enumerate(sorted(range(N), key=lambda y: curlist[y])):
		output[x] = i
	return output


####################################################################################
# analysis
def description_to_nouns(desc):

	desc = desc.replace("/", " ")
	desc = desc.lower().strip()
	res = []

	sentences = cube(desc)
	for s in sentences:
		for entry in s:

			if any(char.isdigit() for char in entry.word):  # filter out any digit
				continue

			word = spell(entry.word)  # spell check and auto correstion
			if 'NOUN' in entry.upos:
				if word in deny_list:
					res.append('FILTERED')
				elif word in stopword_list:
					res.append('FILTERED')
				else:
					res.append(entry.word)
	return res


def count_nouns(D_coords_fixated, curimname, times, thresh=3, apply_group=False):

	parsed_descs = {}
	descs_all = []
	descs = {}
	new_descs = {}

	for t in times:
		descs[t] = []
		new_descs[t] = []
	for elem in D_coords_fixated[curimname]:
		extracted_words = description_to_nouns(elem['description'])
		descs[elem['time']].extend(extracted_words)

	if apply_group:
		concepts, synonym_tokens, r = groupSynonyms(list(filter(lambda x: x != 'FILTERED', descs[500] + descs[3000])))
		_, _, r2 = groupHierarchy(concepts, synonym_tokens)

		replaced_words = []
		new_words = []
		for p in r + r2:
			replaced_words.append(p[0])
			new_words.append(p[1])

		for t in times:
			for w in descs[t]:
				if w in replaced_words:
					idx = replaced_words.index(w)
					new_descs[t].append(new_words[idx])
				else:
					new_descs[t].append(w)

	for t in times:
		descs_all.extend(new_descs[t])

	Counter_all = Counter(descs_all)
	for t in times:
		parsed_descs[t] = {k: v for k, v in Counter(new_descs[t]).items() if Counter_all[k] >= thresh}

	# reformat the above to have dictionary indexed by word
	parsed_descs_across_time = {}
	for t in range(len(times)):
		for n in parsed_descs[times[t]]:
			if n not in parsed_descs_across_time:
				parsed_descs_across_time[n] = np.zeros(len(times))
			parsed_descs_across_time[n][t] = parsed_descs[times[t]][n]

	return parsed_descs, parsed_descs_across_time


def compute_histograms(D_coords_fixated, imnames, category, is_grouping=False, fre_threshold=3):
	D_entropy = {}
	D_histogram = {}
	st.write('Parsing description data')

	progress_bar = st.progress(0)
	for i in range(len(imnames)):
		curimname = imnames[i]

		parsed_descs, parsed_descs_across_time = count_nouns(D_coords_fixated, curimname, times,
															 apply_group=is_grouping, thresh=fre_threshold)
		words_used = list(parsed_descs_across_time.keys())

		D_entropy[curimname] = {}
		D_histogram[curimname] = {'words_used': words_used}

		for t in range(len(times)):
			Ht = [parsed_descs_across_time[k][t] for k in words_used]
			Ht_norm = [elem / float(np.sum(Ht)) for elem in Ht]

			D_histogram[curimname][times[t]] = Ht
			D_entropy[curimname][times[t]] = compute_entropy(Ht_norm)

		progress_bar.progress(float(i + 1) / len(imnames))

	# compute a dataframe for entropy (in the future, would be nice to get rid of dicts entirely)
	Hs = {}
	for t in times:
		Hs[t] = []

	for im in imnames:
		for t in times:
			Hs[t].append(D_entropy[im][t])

	d = {"image": imnames}
	d.update({t: Hs[t] for t in times})

	D_entropy_df = pd.DataFrame(d)
	D_entropy_df.sort_values(by='image', inplace=True)
	D_entropy_df['category'] = category
	return D_histogram, D_entropy, D_entropy_df


def findSynonyms(concepts_list, word):
	for c in concepts_list:
		for synset in wn.synsets(c):
			for lemma in synset.lemma_names():
				if word == lemma:
					return c
	return None


def groupSynonyms(words):
	try:
		counting_res = Counter(words)
	except TypeError:
		words_list = list(itertools.chain.from_iterable(words))
		counting_res = Counter(words_list)

	concepts = {}
	for k, c in counting_res.most_common():
		if k in concepts.keys():
			concepts[k] += c
		else:
			find_syn = findSynonyms(concepts.keys(), k)
			if find_syn is None:
				concepts[k] = c
			else:
				concepts[find_syn] += c

	synonym_tokens = []
	replacements = []
	if type(words[0]) == list:
		for desp in words:
			filted_desp = []
			for t in desp:
				if t in concepts.keys():
					filted_desp.append(t)
				else:
					find_syn = findSynonyms(concepts.keys(), t)
					filted_desp.append(find_syn)
					replacements.append([t, find_syn])
			synonym_tokens.append(filted_desp)
	else:
		for t in words:
			if t in concepts.keys():
				synonym_tokens.append(t)
			else:
				find_syn = findSynonyms(concepts.keys(), t)
				synonym_tokens.append(find_syn)
				replacements.append([t, find_syn])
	return concepts, synonym_tokens, replacements


def distanceToAncestor(word_synset, low_common_set):
	paths = word_synset._shortest_hypernym_paths(low_common_set)
	for k, v in paths.items():
		# using distance to the lowest common ancestor as the distance between the current token
		# and the contral concept of the image
		if k._lemma_names[0] == low_common_set._lemma_names[0]:
			return v


def distanceInHierarchy(a_synset, b_synset):
	if len(a_synset.lowest_common_hypernyms(b_synset)):
		low_common_set = a_synset.lowest_common_hypernyms(b_synset)[0]
	else:
		return MAX_DISTANCE

	d0 = distanceToAncestor(a_synset, low_common_set)
	d1 = distanceToAncestor(b_synset, low_common_set)
	d = d0 + d1
	return d


def isSynonymsInHierarchy(a_synset, b_synset):
	d = distanceInHierarchy(a_synset, b_synset)
	if d < HIER_THRESH:
		return True
	else:
		return False


def groupHierarchy(raw_input_dict, words):
	input_dict = {k: v for k, v in sorted(raw_input_dict.items(), key=lambda item: item[1])}
	concepts = {}

	c_list = list(input_dict.keys())

	while c_list:
		target = c_list.pop()
		if len(wn.synsets(target)):
			target_synset = wn.synsets(target)[0]
		else:  # word outside of wordnet
			# print(f"{bcolors.WARN}Warning: enconter word %s outside wordnet!!{bcolors.END}" % target)
			continue

		concepts[target] = input_dict[target]

		tmp_list = []
		while c_list:
			word = c_list.pop()
			if len(wn.synsets(word)):
				w_synset = wn.synsets(word)[0]
			else:  # word outside of wordnet
				# print(f"{bcolors.WARN}Warning: enconter word %s outside wordnet!!{bcolors.END}" % word)
				continue

			is_synonyms = isSynonymsInHierarchy(w_synset, target_synset)
			if is_synonyms:
				concepts[target] += input_dict[word]
			else:
				tmp_list.append(word)
		c_list = tmp_list

	synonym_tokens = []
	replacements = []
	if type(words[0]) == list:
		for desp in words:
			filted_desp = []
			for t in desp:
				if t in concepts.keys():
					filted_desp.append(t)
				else:
					if len(wn.synsets(t)):
						t_synset = wn.synsets(t)[0]
					else:
						filted_desp.append(t)
						continue
					found_syn = False
					for c in concepts.keys():
						c_synset = wn.synsets(c)[0]
						is_synonyms = isSynonymsInHierarchy(c_synset, t_synset)
						if is_synonyms:
							filted_desp.append(c)
							found_syn = True
							replacements.append([t, c])
							break
					if not found_syn:
						filted_desp.append(t)
			synonym_tokens.append(filted_desp)
	else:
		for t in words:
			if t in concepts.keys():
				synonym_tokens.append(t)
			else:
				if len(wn.synsets(t)):
					t_synset = wn.synsets(t)[0]
				else:
					synonym_tokens.append(t)
					continue
				found_syn = False
				for c in concepts.keys():
					c_synset = wn.synsets(c)[0]
					is_synonyms = isSynonymsInHierarchy(c_synset, t_synset)
					if is_synonyms:
						synonym_tokens.append(c)
						found_syn = True
						replacements.append([t, c])
						break
				if not found_syn:
					synonym_tokens.append(t)
	return concepts, synonym_tokens, replacements


def compute_description_df(D_coords_fixated, curimname, parse_syns=False, parse_hierarchy=False):
	times = []
	texts = []
	tokenses = []
	replacements = {}
	for elem in D_coords_fixated[curimname]:
		if len(elem['description']) > 0:
			times.append(elem['time'])
			texts.append(elem['description'])
			tokenses.append(list(filter(lambda x: x != 'FILTERED', description_to_nouns(elem['description']))))

	if parse_syns:
		if parse_hierarchy:
			concepts, synonym_tokens, rep = groupSynonyms(tokenses)
			replacements['syn'] = rep
			concepts_2, hier_tokens, rep2 = groupHierarchy(concepts, synonym_tokens)
			replacements['hier'] = rep2
			df = pd.DataFrame({"time": times, "description": texts, "tokens": tokenses, "synonyms": synonym_tokens,
							   "hier-synonyms": hier_tokens})
		else:
			concepts, synonym_tokens, rep = groupSynonyms(tokenses)
			replacements['syn'] = rep
			replacements['hier'] = []
			df = pd.DataFrame({"time": times, "description": texts, "tokens": tokenses, "synonyms": synonym_tokens})
	else:
		df = pd.DataFrame({"time": times, "description": texts, "tokens": tokenses})

	df.sort_values(by='time', inplace=True)
	return df, replacements


####################################################################################
# plot
def plot_histogram_words_over_time(H, words_used, filename, axes=None, normalize=False, grouping=False,
								   entropy_value=None):
	if grouping:
		grouped_words = []
		grouped_H = {}
		grouped_H[0] = []
		grouped_H[1] = []
		others = [0, 0]
		for i in range(len(words_used)):
			if H[0][i] > 1 or H[1][i] > 1:
				grouped_words.append(words_used[i])
				grouped_H[0].append(H[0][i])
				grouped_H[1].append(H[1][i])
			else:
				others[0] += H[0][i]
				others[1] += H[1][i]

		grouped_words.append("[other]")
		grouped_H[0].append(others[0])
		grouped_H[1].append(others[1])
		words_used = grouped_words
		H = grouped_H

	if normalize:
		H[0] = np.asarray(H[0]).astype(np.float)
		H[1] = np.asarray(H[1]).astype(np.float)
		H[0] *= (1.0 / np.sum(H[0]))
		H[1] *= (1.0 / np.sum(H[1]))
		H[0] = H[0].tolist()
		H[1] = H[1].tolist()

	if not axes:
		fig, axes = plt.subplots(ncols=2, sharey=True)

	nwords = len(words_used)
	r1 = np.arange(nwords)
	plt.rcParams.update({'font.size': 15})

	axes[0].barh(r1, H[0], align='center', color='#1F77B4')
	axes[0].set(title='0.5 sec\n($H_{0.5}$: %.2f)' % entropy_value[500])
	axes[1].barh(r1, H[1], align='center', color='#FF7F0E')
	axes[1].set(title='3 sec\n($H_{3}$: %.2f)' % entropy_value[3000])
	axes[0].set(yticks=r1, yticklabels=words_used)
	axes[0].yaxis.tick_right()
	maxv = np.max([H[0], H[1]])
	axes[0].set_xlim(0, maxv)
	axes[1].set_xlim(0, maxv)
	axes[0].invert_xaxis()
	axes[0].spines['right'].set_visible(False)
	axes[0].spines['top'].set_visible(False)
	axes[0].spines['left'].set_visible(False)
	axes[0].tick_params(length=0)
	axes[1].spines['right'].set_visible(False)
	axes[1].spines['top'].set_visible(False)
	axes[1].spines['left'].set_visible(False)
	axes[1].tick_params(length=0)

	fig.suptitle('Relative frequency of words across descriptions', verticalalignment='bottom')
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.40)
	plt.gca().invert_yaxis()
	placeholder = st.empty()

	if display_mode == 'Single image':
		save_pdf = st.button('Save histogram to PDF')
		if save_pdf:
			basename = os.path.basename(filename)
			i = basename.index('.')
			filename = basename[:i] + '.pdf'
			plt.savefig(filename, bbox_inches='tight')
			st.write('file saved to streamlit folder as "' + filename + '"')

	placeholder.pyplot()
	plt.close()


def print_N_images_by_inds(imnames, inds, N_top=10):
	for i in range(N_top):
		curid = inds[i]
		curimname = imnames[curid]

		st.write(os.path.basename(curimname))
		st.image(Image.open(curimname).resize((256, 256)))
		st.write('Entropy at 500: %2.2f, Entropy at 3000: %2.2f' % (entropy_500[curid], entropy_3000[curid]))
		st.markdown('---')


def print_N_images_by_inds_with_scores(imnames, inds, N_top=10, score_type=None, score_arry=None, add_info=None):
	for i in range(N_top):
		curid = inds[i]
		curimname = imnames[curid]

		st.subheader('Image: ' + os.path.basename(curimname))
		if score_type and score_arry:
			st.write('%s score: %2.2f' % (score_type, score_arry[curimname]))
			if add_info:
				st.write(add_info[curimname])
		st.image(Image.open(curimname).resize((256, 256)))


####################################################################################
st.title("Image Indeterminacy Task")

stemmer = initNLTK()
stopword_list = loadStopwords(stopword_file_path)

category = st.sidebar.selectbox('Image category', options=study_names + ["All"])
if category == "All":
	studyname = 'All'
else:
	studyname = '%s_30_leniant' % category

FREQUENCY_THRESH = st.sidebar.slider('Threshold for frequency (suggested value: 1)', 1, 3, FREQUENCY_THRESH)
st.sidebar.markdown('---')

recompute_histograms = st.sidebar.button("Recompute histograms")
st.sidebar.markdown(
	"_Press this button when you've modified the histogramming code during an interactive session, otherwise they'll be loaded from a file_")

st.sidebar.markdown('---')


####################################################################################
if recompute_histograms or not os.path.isfile(histogram_picklefile_gf % (category, FREQUENCY_THRESH)):
	if category == "All":
		D_coords_fixated, D_histogram, D_entropy, D_entropy_df = \
			merge_histograms(histogram_picklefile_gf)
		imnames = list(set(D_coords_fixated.keys()))
		pickle.dump([D_coords_fixated, D_histogram, D_entropy, D_entropy_df],
					open(histogram_picklefile_gf % (category, FREQUENCY_THRESH), "wb"))
	else:
		D_coords_fixated = get_raw_data()
		imnames = list(set(D_coords_fixated.keys()))
		cube = Cube(verbose=True)
		cube.load('en')
		start = time.time()
		D_histogram, D_entropy, D_entropy_df = compute_histograms(D_coords_fixated, imnames, category,
																  is_grouping=True, fre_threshold=FREQUENCY_THRESH)
		print(time.time()-start)
		pickle.dump([D_coords_fixated, D_histogram, D_entropy, D_entropy_df],
					open(histogram_picklefile_gf % (category, FREQUENCY_THRESH), "wb"))
		print(histogram_picklefile_gf % (category, FREQUENCY_THRESH))
else:
	with open(histogram_picklefile_gf % (category, FREQUENCY_THRESH), 'rb') as file:
		p = pickle.load(file)
		print("load file ", file)

	D_coords_fixated = p[0]
	D_histogram = p[1]
	D_entropy = p[2]
	D_entropy_df = p[3]
	imnames = list(set(D_coords_fixated.keys()))

imbasenames = [os.path.basename(imname) for imname in imnames]

####################################################################################
display_mode = st.sidebar.selectbox('Display mode', options=['Single image', 'First N images', 'Scatterplot',
															 'Sorted images', 'None'])

if display_mode == 'First N images':  # show_multiple:
	num_to_show = st.sidebar.slider("Num to show", value=10, min_value=1, max_value=len(imnames), step=1)
	show_range = range(num_to_show)
elif display_mode == 'Single image':
	basenameToIndex = {imbasenames[i]: i for i in range(len(imbasenames))}
	image_to_show = st.sidebar.selectbox('Image to show', options=sorted(imbasenames))
	index_to_show = basenameToIndex[image_to_show]
	show_range = [index_to_show]
else:  # scatterplot mode
	show_range = []

####################################################################################
if display_mode == "Single image" or display_mode == "First N images":
	if display_mode == "Single image":
		show_descriptions = st.sidebar.checkbox('Show descriptions', value=False)
		show_description_table = st.sidebar.checkbox('Show description table (takes a few seconds)', value=False)

		if show_description_table:
			show_grouping_details = st.sidebar.checkbox('Show synonym pairs', value=False)

	show_histograms = st.sidebar.checkbox('Plot histograms of nouns', value=True)
	if show_histograms:
		normalize_histogram = st.sidebar.checkbox("Normalise histogram", value=True)
		group_infrequent_words = st.sidebar.checkbox("Group infrequent words", value=True)

	for i in show_range:
		curimname = imnames[i]
		st.subheader('Image %d / %d : %s' % (i, len(imnames), os.path.basename(curimname)))
		st.write('Category **%s**, Entropy H_500: **%.2f**, H_3000: **%.2f**' % (
			D_entropy_df.loc[D_entropy_df['image'] == curimname, 'category'].iloc[0],
			D_entropy[curimname][500], D_entropy[curimname][3000]))
		st.image(Image.open(curimname).resize((256, 256)))

		if display_mode == "Single image":
			# --------- print all raw descriptions ---------
			if show_descriptions:
				if leave_out_FILTERED:
					try:
						st.markdown('**Filtered out following number of words from descriptions: **' + str(rem))
					except NameError:
						rem = {500: 0.0, 3000: 0.0}
						st.markdown('**Filtered out following number of words from descriptions: **' + str(rem))

				all_descriptions = ["point five second"]
				for time in times:
					D_selected = [elem for elem in D_coords_fixated[curimname] if elem['time'] == time]
					descriptions = [elem['description'] for elem in D_selected]
					desc_lengths = np.mean([len(desc) for desc in descriptions])

					st.write('\n**Time: %d, Avg. length of description: %2.2f, Descriptions:** %s' % (
						time, desc_lengths, descriptions))
					all_descriptions += descriptions
					if time == 500:
						all_descriptions.append("three three three seconds")

			# --------- print description table ---------
			if show_description_table:
				cube = Cube(verbose=True)
				cube.load('en')

				st.subheader('All descriptions')
				description_df, replacement_dict = compute_description_df(D_coords_fixated, curimname,
																		  parse_syns=True,
																		  parse_hierarchy=True)
				st.write(description_df)
				if show_grouping_details:
					if len(replacement_dict.keys()) >= 1:
						st.write("**Synonyms:**")
						for item in replacement_dict['syn']:
							st.write("%s replaced by %s" % (item[0], item[1]))

					if len(replacement_dict.keys()) == 2:
						st.write("**Hierarchy:**")
						for item in replacement_dict['hier']:
							st.write("%s replaced by %s" % (item[0], item[1]))

		# --------- plot histogram of words used in descriptions ---------
		if show_histograms:
			st.subheader('Word usage histogram')
			words_used = D_histogram[curimname]['words_used'].copy()
			H = [D_histogram[curimname][500].copy(), D_histogram[curimname][3000].copy()]

			if leave_out_FILTERED:
				try:
					ind = words_used.index('FILTERED')
					words_used.pop(ind)
					rem = {500: H[0][ind], 3000: H[1][ind]}
					H[0].pop(ind)
					H[1].pop(ind)
				except ValueError:
					pass
			plot_histogram_words_over_time(H, words_used, curimname, normalize=normalize_histogram,
										   grouping=group_infrequent_words,
										   entropy_value=D_entropy[curimname])

####################################################################################

if display_mode == 'Scatterplot':

	H_500 = D_entropy_df[500]
	H_3000 = D_entropy_df[3000]
	H_difference = H_500 - H_3000

	label_dict = {'H_500': 'H_0.5', 'H_3000': 'H_3', 'H_500 - H_3000': 'H_0.5-H_3'}

	h_values = {'H_500': H_500, 'H_3000': H_3000, 'H_500 - H_3000': H_difference}

	x_axis = st.sidebar.selectbox('X axis', options=list(h_values))
	y_axis = st.sidebar.selectbox('Y axis', options=list(h_values))

	h_values['category'] = D_entropy_df['category']
	h_values['image'] = [os.path.basename(imname) for imname in D_entropy_df['image']]
	scatter_df = pd.DataFrame(data=h_values)
	discard_cat_label = False
	if category == "All":
		st.sidebar.markdown("Uncheck the catagory to exclude the data in the plot")
		show_abstract = st.sidebar.checkbox('Show data from abstract', value=True)
		show_abstract_flat = st.sidebar.checkbox('Show data from abstract flat', value=True)
		show_dichotomour = st.sidebar.checkbox('Show data from dichotomour', value=True)
		show_indeterminate = st.sidebar.checkbox('Show data from indeterminate', value=True)
		show_recognizable = st.sidebar.checkbox('Show data from recognizable', value=True)

		plot_catagories = []
		if show_abstract:
			plot_catagories.append("Abstract")
		if show_abstract_flat:
			plot_catagories.append("AbstractFlat")
		if show_dichotomour:
			plot_catagories.append("Dichotomous")
		if show_indeterminate:
			plot_catagories.append("Indeterminate")
		if show_recognizable:
			plot_catagories.append("Recognizable")
		plot_df = scatter_df.loc[scatter_df['category'].isin(plot_catagories)]
	else:
		plot_df = scatter_df

	scatterplot = alt.Chart(plot_df).mark_point().encode(
		x=alt.X('%s:Q' % x_axis, title=label_dict[x_axis]),
		y=alt.Y('%s:Q' % y_axis, title=label_dict[y_axis]),
		color='category:N',
		tooltip='image:N'
	)
	st.altair_chart(scatterplot, use_container_width=False)


####################################################################################

if display_mode == 'Sorted images':
	show_sorted_images = st.sidebar.checkbox('Show images sorted by entropy', value=False)
	show_sorted_images_by_H_diff = st.sidebar.checkbox('Show images sorted by entropy difference', value=False)
	show_sorted_images_by_H_3000 = st.sidebar.checkbox('Show images sorted by H3000', value=False)

	H_500_values = np.asarray([D_entropy[curimname][500] for curimname in imnames])
	H_3000_values = np.asarray([D_entropy[curimname][3000] for curimname in imnames])

	max_H_500 = float(math.ceil(np.max(H_500_values)))
	max_H_3000 = float(math.ceil(np.max(H_3000_values)))
	min_H_500 = float(math.floor(np.min(H_500_values)))
	min_H_3000 = float(math.floor(np.min(H_3000_values)))
	thresh_500 = st.sidebar.slider('Lower threshold for H_500 values', min_H_500,
								   max_H_500, (min_H_500, max_H_500), step=0.1)
	thresh_3000 = st.sidebar.slider('Lower threshold for H_3000 values', min_H_3000,
									max_H_3000, (min_H_3000, max_H_3000), step=0.1)

	H_500_idx = np.where(np.logical_and(thresh_500[0] <= H_500_values, H_500_values <= thresh_500[1]))[0]
	H_3000_idx = np.where(np.logical_and(thresh_3000[0] <= H_3000_values, H_3000_values <= thresh_3000[1]))[0]
	H_idx = np.intersect1d(H_500_idx, H_3000_idx)
	thresh_names = np.asarray(imnames)[H_idx]

	if show_sorted_images:
		st.markdown('---')
		thresh_top_N = st.sidebar.slider('Show top n examples', 0, len(thresh_names), 1)
		entropy_500 = [D_entropy[curimname][500] for curimname in imnames]
		entropy_500 = np.asarray(entropy_500)[H_idx]
		entropy_3000 = [D_entropy[curimname][3000] for curimname in imnames]
		entropy_3000 = np.asarray(entropy_3000)[H_idx]

		ims_with_high_500_entropy = get_elem_ranks(entropy_500)  # larger rank means higher entropy
		ims_with_low_500_entropy = get_elem_ranks([-elem for elem in entropy_500])  # larger rank means lower entropy
		ims_with_high_3000_entropy = get_elem_ranks(entropy_3000)
		ims_with_low_3000_entropy = get_elem_ranks([-elem for elem in entropy_3000])
		N = len(ims_with_high_500_entropy)

		st.header('Confusing images (high entropy at 500 and 3000)')
		ims_confusing = [ims_with_high_500_entropy[i] + ims_with_high_3000_entropy[i] for i in range(N)]
		# higher rank means higher entropy for 500 and higher entropy for 3000
		ids_ims_confusing = np.argsort(ims_confusing)[::-1]  # first images are classified as most confusing
		print_N_images_by_inds(thresh_names, ids_ims_confusing, N_top=thresh_top_N)

		st.header('Determinate images (low entropy at 500 and 3000)')
		ims_determinate = [ims_with_low_500_entropy[i] + ims_with_low_3000_entropy[i] for i in range(N)]
		ids_ims_determinate = np.argsort(ims_determinate)[::-1]  # first images are classified as most determinate
		print_N_images_by_inds(thresh_names, ids_ims_determinate, N_top=thresh_top_N)

		st.header('Indeterminate images (low entropy at 500, high entropy 3000)')
		ims_indeterminate = [ims_with_low_500_entropy[i] + ims_with_high_3000_entropy[i] for i in range(N)]
		ids_ims_indeterminate = np.argsort(ims_indeterminate)[::-1]  # first images are classified as most indeterminate
		print_N_images_by_inds(thresh_names, ids_ims_indeterminate, N_top=thresh_top_N)

		st.header('Hidden (aha) images  (high entropy at 500, low entropy 3000)')
		ims_hidden = [ims_with_high_500_entropy[i] + ims_with_low_3000_entropy[i] for i in range(N)]
		ids_ims_hidden = np.argsort(ims_hidden)[::-1]  # first images are classified as most hidden
		print_N_images_by_inds(thresh_names, ids_ims_hidden, N_top=thresh_top_N)

	if show_sorted_images_by_H_diff:
		st.markdown('---')
		st.header("Images sorted by weighted entropy difference")
		H_500_values = np.asarray([D_entropy[curimname][500] for curimname in imnames])
		H_3000_values = np.asarray([D_entropy[curimname][3000] for curimname in imnames])

		H_500_values = H_500_values[H_idx]
		H_3000_values = H_3000_values[H_idx]

		diff = H_500_values - H_3000_values
		sorted_idx = np.argsort(diff)
		diff_arr = {}
		add_arr = {}
		thresh_names = []
		for s_idx in sorted_idx:
			idx = H_idx[s_idx]
			sel_imname = imnames[idx]
			diff_arr[sel_imname] = diff[s_idx]
			add_arr[sel_imname] = "H_500: %.2f, H_3000: %.2f" % (H_500_values[s_idx], H_3000_values[s_idx])
			thresh_names.append(sel_imname)
		print_N_images_by_inds_with_scores(thresh_names, np.arange(len(diff_arr)), N_top=len(thresh_names),
											   score_type="Entropy difference", score_arry=diff_arr, add_info=add_arr)

	if show_sorted_images_by_H_3000:
		st.markdown('---')
		st.header("Images sorted by weighted H_3000")
		H_3000_values = np.asarray([D_entropy[curimname][3000] for curimname in imnames])

		H_3000_values = H_3000_values[H_idx]

		sorted_idx = np.argsort(H_3000_values)
		diff_arr = {}
		add_arr = {}
		thresh_names = []
		for s_idx in sorted_idx:
			idx = H_idx[s_idx]
			sel_imname = imnames[idx]
			diff_arr[sel_imname] = H_3000_values[s_idx]
			add_arr[sel_imname] = "H_3000: %.2f" % (H_3000_values[s_idx])
			thresh_names.append(sel_imname)
		print_N_images_by_inds_with_scores(thresh_names, np.arange(len(diff_arr)), N_top=len(thresh_names),
											   score_type="Entropy difference", score_arry=diff_arr, add_info=add_arr)


