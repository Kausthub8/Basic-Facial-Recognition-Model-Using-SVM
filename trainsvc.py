def plot_svc_decision_function(model, ax=None, plot_support=True):

  if ax is None:
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()


    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    p = model.decision_function(xy).reshape(X.shape)


  ax.contour(X, Y, p, color='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])


  if plot_support:

    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=100, linewidth=1, facecolors='red');


  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
