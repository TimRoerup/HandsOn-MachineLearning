import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(1000)

class Drawer:
    def __init__(self, samples_1D=1000, samples_2D=10000, input=None, bins=30,
                 mean_2D=[0.5, -0.2], mean_1D=1, sigma_2D=[[2.0, 0.3], [0.3, 0.5]],
                 sigma_1D=0.2, plot_flag=True):
        self.samples_1D = samples_1D
        self.samples_2D = samples_2D
        self.mean_1D = mean_1D
        self.sigma_1D = sigma_1D
        self.bins = bins
        self.mean_2D = mean_2D
        self.sigma_2D = sigma_2D
        self.input = input
        self.plot_flag =plot_flag

    def draw_2D(self):
        samples = np.random.normal(loc=self.mean_1D, scale=self.sigma_1D, size=self.samples_1D)
        if self.input is not None:
            samples = self.input
        if self.plot_flag:
            fig, axs = plt.subplots(2)
            axs[0].hist(samples, bins=self.bins)
            x = np.linspace(self.mean_1D - 3*self.sigma_1D, self.mean_1D + 3*self.sigma_1D, 200)
            axs[1].plot(x, stats.norm.pdf(x, self.mean_1D, self.sigma_1D))

            plt.show()

        return samples

    def draw_3D(self):
        samples = np.random.multivariate_normal(mean=self.mean_2D, cov=self.sigma_2D, size=self.samples_2D)
        if self.input is not None:
            samples = self.input

        if self.plot_flag:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            x = samples[:, :1].ravel()
            y = samples[:, 1:].ravel()
            hist, xedges, yedges = np.histogram2d(x, y, bins=self.bins)
            # Construct arrays for the anchor positions of the 16 bars.
            xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = 0

            # Construct arrays with the dimensions for the 16 bars.
            dx = dy = 0.5 * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')


            plt.show()

        return samples

class GapStatistic:
    def __init__(self, B=50, max_k=10):
        self.B = B
        self.max_k = max_k
        self.wcds_uniform = {}
        self.wcds_interest = {}
        self.uniforms_std = {}
        self.kmeans = None
        self.flag = None

        for k in range(1, self.max_k):
            self.wcds_uniform[str(k)] = []
            self.wcds_interest[str(k)] = []
            self.uniforms_std[str(k)] = []

    def within_cluster_distance(self, array, centers, k):
        """Calculates the overall within cluster distance.

            :param array: Samples and labels. [[x_11,...,x_n1,label_1], [x_12,...,x_n2,label_n]]
            :param centers: K centers of kmeans
            :return: Scalar representing the log within cluster distance.
            """
        wcd = 0
        for row in array:
            wcd += np.linalg.norm(row[:-1] - centers[int(row[-1:].item())]) ** 2

        if self.flag:
            self.wcds_uniform[str(k)].append(np.log(wcd))
            self.uniforms_std[str(k)].append(np.std(uniform[:, :-1]))
        else:
            self.wcds_interest[str(k)].append(np.log(wcd))

    def apply_kmeans(self, k, data, flag):
        """

        :param k: nr. of clusters
        :param data: uniform reference data or the data of interest
        :param flag: Boolean. True-> data is a uniform distribution; False->data is data of interest
        :return:
        """
        self.flag = flag
        self.kmeans = KMeans(n_clusters=k).fit(data)

    def select_k(self, gaps_uni, gaps_interest, uniforms_std):
        gap = gaps_uni - gaps_interest
        for k in range(len(gap)):
            if gap[k] > gap[k+1] - uniforms_std[k] * np.sqrt(1 + (1/self.B)):
                return k + 1
            else:
                pass



if __name__ == '__main__':
    # Create data to cluster
    drawer_1 = Drawer(samples_2D=100, mean_2D=[3,3], plot_flag=False)
    drawer_2 = Drawer(samples_2D=100, mean_2D=[-2,2], plot_flag=False)
    drawer_3 = Drawer(samples_2D=100, mean_2D=[-2,-8], plot_flag=False)
    s1 = drawer_1.draw_3D()
    s2 = drawer_2.draw_3D()
    s3 = drawer_3.draw_3D()

    samples = np.append(np.append(s1,s2,axis=0),s3,axis=0)

    fig, axs = plt.subplots(4)

    # Plot original data
    axs[0].scatter(s1[:, :1].ravel(), s1[:, 1:].ravel(), color='r', alpha=0.3)
    axs[0].scatter(s2[:, :1].ravel(), s2[:, 1:].ravel(), color='g', alpha=0.3)
    axs[0].scatter(s3[:, :1].ravel(), s3[:, 1:].ravel(), color='b', alpha=0.3)

    gap = GapStatistic()

    for k in range(1, gap.max_k):
        gap.apply_kmeans(k=k, data=samples, flag=False)
        clustered_data = np.append(samples, np.reshape(gap.kmeans.labels_, (-1,1)), axis=1)
        gap.within_cluster_distance(array=clustered_data, centers=gap.kmeans.cluster_centers_, k=k)

        for b in range(gap.B):
            uniform = np.random.uniform(low=0, high=1, size=(100, 2))
            gap.apply_kmeans(k=k, data=uniform, flag=True)
            clustered_data = np.append(uniform, np.reshape(gap.kmeans.labels_, (-1, 1)), axis=1)
            gap.within_cluster_distance(array=clustered_data, centers=gap.kmeans.cluster_centers_, k=k)


    gaps_uniform = np.sum(list(gap.wcds_uniform.values()), axis=1) / len(list(gap.wcds_uniform.values())[0])
    uniforms_std = np.sum(list(gap.uniforms_std.values()), axis=1) / len(list(gap.uniforms_std.values())[0])
    gaps_interest = np.ravel(list(gap.wcds_interest.values()))

    # Output best k according to gap statistic
    best_k = gap.select_k(gaps_uniform, gaps_interest, uniforms_std)
    print('Best k:', best_k)

    # Do final kMeans clustering
    kmeans = KMeans(n_clusters=best_k).fit(samples)
    clustered_data = np.append(samples, np.reshape(kmeans.labels_, (-1, 1)), axis=1)

    # Plot clustered data in different color per cluster
    for i in range(best_k):
        colors = ['r', 'b', 'g', 'c', 'm']
        axs[1].scatter(clustered_data[clustered_data[:,-1] == i][:, 0].ravel(),
                       clustered_data[clustered_data[:,-1] == i][:, 1].ravel(), color=colors[i], alpha=0.3)

    # Plot k over gap
    axs[2].errorbar(np.linspace(1, len(gaps_interest), len(gaps_interest)), gaps_uniform - gaps_interest, uniforms_std)
    # Plot within cluster distance over k of original distribution
    axs[3].plot(np.linspace(1, len(gaps_interest), len(gaps_interest)), gaps_interest)
    plt.show()