import numpy as np
import matplotlib.pyplot as plt

list = ['results/acin_hallway_cardboard',
        'results/acin_hallway_cigarettes',
        'results/acin_hallway_clean',
        'results/acin_hallway_liquids',
        'results/acin_my_office_cardboard',
        'results/acin_my_office_cigarettes',
        'results/acin_my_office_liquids_light']


n = 200
count_true_pos = np.zeros(n)#[1 for i in r]
count_false_pos = np.zeros(n)#[1 for i in r]
count_true_neg = np.zeros(n)#[0 for i in r]
count_false_neg = np.zeros(n)#[0 for i in r]

for file in list:
    count_true_pos += np.load(file + '_count_true_pos.npy')
    count_false_pos += np.load(file + '_count_false_pos.npy')
    count_true_neg += np.load(file + '_count_true_neg.npy')
    count_false_neg += np.load(file + '_count_false_neg.npy')

tpr_acin = np.divide(count_true_pos, count_true_pos + count_false_neg)
fpr_acin = np.divide(count_false_pos, count_false_pos + count_true_neg)

count_true_pos = np.load('results/bormann' + '_count_true_pos.npy')
count_false_pos = np.load('results/bormann' + '_count_false_pos.npy')
count_true_neg = np.load('results/bormann' + '_count_true_neg.npy')
count_false_neg = np.load('results/bormann' + '_count_false_neg.npy')
tpr_bormann = np.divide(count_true_pos, count_true_pos + count_false_neg)
fpr_bormann = np.divide(count_false_pos, count_false_pos + count_true_neg)

plt.plot(fpr_acin, tpr_acin, fpr_bormann, tpr_bormann)#, [0, 1], [0, 1])
plt.ylabel('True positive rate (tpr)')
plt.xlabel('False positive rate (fpr)')
plt.legend(['ACIN Dataset', 'IPA Dataset'])
plt.title('Existing Datasets')
plt.show()


count_true_pos = np.load('results/flobot_lyon' + '_count_true_pos.npy')
count_false_pos = np.load('results/flobot_lyon' + '_count_false_pos.npy')
count_true_neg = np.load('results/flobot_lyon' + '_count_true_neg.npy')
count_false_neg = np.load('results/flobot_lyon' + '_count_false_neg.npy')
tpr_lyon = np.divide(count_true_pos, count_true_pos + count_false_neg)
fpr_lyon = np.divide(count_false_pos, count_false_pos + count_true_neg)


count_true_pos = np.load('results/flobot_carugate' + '_count_true_pos.npy')
count_false_pos = np.load('results/flobot_carugate' + '_count_false_pos.npy')
count_true_neg = np.load('results/flobot_carugate' + '_count_true_neg.npy')
count_false_neg = np.load('results/flobot_carugate' + '_count_false_neg.npy')
tpr_carugate = np.divide(count_true_pos, count_true_pos + count_false_neg)
fpr_carugate = np.divide(count_false_pos, count_false_pos + count_true_neg)


plt.plot(fpr_carugate, tpr_carugate, fpr_lyon, tpr_lyon)#, [0, 1], [0, 1])
plt.ylabel('True positive rate (tpr)')
plt.xlabel('False positive rate (fpr)')
plt.title('FLOBOT Dataset')
plt.legend(['Supermarket Carugate', 'Airport Lyon'])
plt.show()