
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_graphtask(G, mappingN, Gedges=None, font_size=10):
	''' '''
	fig = plt.figure(figsize=(8,3))
	plt.subplot(1,2,1)
	nx.draw(G, with_labels=True,  labels=mappingN, node_size=100, 
			node_color='lightgreen', font_size=font_size, font_color='k')

	if Gedges is not None: 
		plt.subplot(1,2,2)
		plt.imshow(Gedges,cmap='binary')
		plt.title('Edges')

	plt.tight_layout()

def parse_dist_accs(frame, dist_id):
    Li, Lb = [], []
    scorei, scoreb = [], []
    for row_idx, row in frame.iterrows():
        if row['task'] == 'I':
            Li.append(row['L2'])
            scorei.append(row['scores'][dist_id])
        elif row['task'] == 'B':
            Lb.append(row['L2'])
            scoreb.append(row['scores'][dist_id])
    
    Li, Lb = np.array(Li), np.array(Lb)
    scorei, scoreb = np.array(scorei), np.array(scoreb)
    return Li, Lb, scorei, scoreb

def plot_results(r_frame):
    dists_l = [1,2,3]
    plt.figure(figsize=(13,3))
    for i in dists_l:
        plt.subplot(1, len(dists_l), i)
        Li, Lb, scorei, scoreb = parse_dist_accs(r_frame, dist_id=i)
        # b inter ,, r blocked
        plt.scatter(Li, scorei, color='teal', alpha=.5)
        plt.scatter(Lb, scoreb, color='r', alpha=.5)

        #find line of best fit
        ai, bi = np.polyfit(Li, scorei, 1)
        ab, bb = np.polyfit(Lb, scoreb, 1)
        plt.plot(Li, ai*Li+bi, color='teal', linewidth=3, label='Intermixed')
        plt.plot(Lb, ab*Lb+bb, color='r', linewidth=3, label='Blocked')
        
        plt.title(f'DistDiff: {i}', size=20)

        if i == 1:
            plt.ylabel('Judgement Accuracy', size=18)
            plt.yticks([0, 25, 50, 75, 100], size=15)
            plt.legend()
        else:
            plt.yticks([])
            

        plt.xlabel('Layer 2 width', size=18)
        plt.ylim(0,100)
        plt.xticks([6, 9, 12, 15, 18], size=15)