import numpy
import time
import logging
import scipy.sparse
import argparse
from .anewton import *
from .utils import *
from utils import show_matrix, check_sparse, matmul, FPR, TP, TPR

# def parse_args():
#     parser = argparse.ArgumentParser(description = 'Collective Matrix Factorization')
#     parser.add_argument('--train' , type = str, default = '', help = 'Training file')
#     parser.add_argument('--test' , type = str, default = '', help = 'Testing file')
#     parser.add_argument('--user' , type = str, default = '', help = 'User features file')
#     parser.add_argument('--item' , type = str, default = '', help = 'Item features file')
#     parser.add_argument('--out', type = str, default = 'out.txt', help = 'File where fianl result will be saved')

#     parser.add_argument('--link' , type = str, default = 'log_dense', help = 'link function for feature relations (dense or log_dense)')

#     parser.add_argument('--alphas' , type = str, default = '0.4-0.3-0.3', help = 'Alpha in [0, 1] weights the relative importance of relations')
#     parser.add_argument('--k', type = int, default = 8, help = 'Dimension of latent fectors')
#     parser.add_argument('--reg', type = float, default = 0.1, help = 'Regularization for latent facotrs')
#     parser.add_argument('--lr', type = float, default = 0.1, help = 'Initial learning rate for training')

#     parser.add_argument('--iter', type = int, default = 10, help = 'Max training iteration')
#     parser.add_argument('--tol', type = float, default = 0, help = 'Tolerant for change in training loss')
#     parser.add_argument('--verbose', type = int, default = 1, help = 'Verbose or not (1 for INFO, 0 for WARNING)')

#     parser.add_argument('--boolean', type = int, default = 0)

#     return parser.parse_args()

def learn(Xs, Xstst, rc_schema, modes, alphas, K, reg, learn_rate, max_iter, tol, logger):
    assert(rc_schema.shape[0] == len(Xs) and rc_schema.shape[1] == 2) # schema match data
    assert(numpy.all(rc_schema[:, 0] != rc_schema[:, 1])) # should not have symmetric relations
    assert(rc_schema.shape[0] == len(alphas))
    assert(rc_schema.shape[0] == len(modes))
    check_modes(modes) 

    Xts = [None] * len(Xs)
    for i in range(len(Xs)):
        if Xs[i] is not None:
            Xts[i] = scipy.sparse.csc_matrix(Xs[i].T) # Transpose
            Xs[i] = scipy.sparse.csc_matrix(Xs[i]) # no Transpose
        if Xstst[i] is not None:
            Xstst[i] = scipy.sparse.csc_matrix(Xstst[i])

    [S, Ns] = get_config(Xs, rc_schema)

    # randomly initialize factor matrices with small values
    Us = [None] * S
    for i in range(S):
        Us[i] = numpy.random.rand(Ns[i], K) * numpy.sqrt(1/K)  # so initial prediction will be in [0, 5]

    Ys = predict(Us, Xs, rc_schema, modes)
    prev_loss = loss(Us, Xs, Ys, rc_schema, modes, alphas, reg, S)
    i = 0
    while i < max_iter:
        i += 1
        tic = time.time()

        # training        
        for t in range(S): # update factors for entity t
            newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t)
        
        # evaluation
        Ys = predict(Us, Xs, rc_schema, modes)
        training_loss = loss(Us, Xs, Ys, rc_schema, modes, alphas, reg, S)
        train_rmse = RMSE(Xs[0], Ys[0])
        change_rate = (training_loss-prev_loss)/prev_loss * 100
        prev_loss = training_loss
        
        Ystst = predict(Us, Xstst, rc_schema, modes)
        test_rmse = RMSE(Xstst[0], Ystst[0])

        # plot
        if i % 20 == 0:
            # for i in range(len(Xs)): # i = 1
            #     if Xs[i] is None:
            #         continue
                
                # X = Xs[i]
            U = Us[rc_schema[0, 0]]
            V = Us[rc_schema[0, 1]].T
            Z = Us[rc_schema[1, 0]]

                # X_pred = sigmoid(U @ V.T)
                # Y_pred = sigmoid(Z @ V.T)
                # X_pred = Ys[i]

            settings = [(check_sparse(Ys[0], sparse=False), [2, 1], "X"),
                        (check_sparse(Xs[0], sparse=False), [2, 2], "X, gt"),
                        (check_sparse(Ys[1].T, sparse=False), [0, 1], "Y"),
                        (check_sparse(Xs[1].T, sparse=False), [0, 2], "Y, gt"),
                        (check_sparse(U, sparse=False), [2, 0], "U"),
                        (check_sparse(V, sparse=False), [1, 1], "V"),
                        (check_sparse(Z, sparse=False), [0, 0], "Z")
                        ]
            show_matrix(settings)


        toc = time.time()
        logger.info('Iter {}/{}. Time: {:.1f}'.format(i, max_iter, toc - tic))
        logger.info('Training Loss: {:.1f} (change {:.2f}%). Training RMSE: {:.2f}. Testing RMSE: {:.2f}'.format(training_loss, change_rate, train_rmse, test_rmse))

        train_gt = (Xs[0] > 0.5) * 1
        train_pd = (Ys[0] > 0.5) * 1
        # train_ppv = PPV(train_gt, train_pd)
        test_gt = (Xstst[0] > 0.5) * 1
        test_pd = (Ystst[0] > 0.5) * 1
        # test_ppv = PPV(test_gt, test_pd)
        # print(Xs[0].min(), Xs[0].max(), Ys[0].min(), Ys[0].max())

        logger.info('Training TPR: {:.2f}. Testing TPR: {:.2f}'.format(TPR(train_gt, train_pd), TPR(test_gt, test_pd)))
        logger.info('Training FPR: {:.2f}. Testing FPR: {:.2f}'.format(FPR(train_gt, train_pd), FPR(test_gt, test_pd)))
        # logger.info('Training precision: {:.2f}. Testing precision: {:.2f}'.format(train_ppv, test_ppv))
    
        # early stop
        if tol!=0 and i!=1 and change_rate > -tol :
            break

    return Us

def loss(Us, Xs, Ys, rc_schema, modes, alphas, reg, num_entities):
	'''
	Calculate objective loss
	See page 4: Generalizing to Arbitrary Schemas
	'''
	assert(rc_schema.shape[0] == len(Xs) and rc_schema.shape[1] == 2)

	res = 0
	num_relation = len(Xs)
	# computing regularization for each latent factor
	for i in range(num_entities):
		for j in range(num_relation):
			if rc_schema[j, 0]==i or rc_schema[j, 1]==i:
				res += alphas[j] * reg * numpy.linalg.norm(Us[i].flat) / 2 # l2 norm

	# computing loss for each relation
	for j in range(num_relation):     
		alpha_j = alphas[j]
		if Xs[j] is None or Ys[j] is None or alpha_j == 0:
			continue

		# X = scipy.sparse.csc_matrix(Xs[j])
		# Y = scipy.sparse.csc_matrix(Ys[j])
		X = Xs[j]
		Y = Ys[j]
		
		if modes[j] == 'sparse':
			assert( X.size == Y.size )
			res += alpha_j * numpy.sum(pow(X.data - Y.data, 2))

		elif modes[j] == 'dense' or modes[j] == 'log_dense':
			assert( numpy.all(Y.shape == X.shape) )
			res += alpha_j * numpy.sum(pow(X.toarray() - Y.toarray(), 2))

	return res

def predict(Us, Xs, rc_schema, modes):
    '''
    see page 3: RELATIONAL SCHEMAS
    return a list of csc_matrix
    '''
    Ys = []
    for i in range(len(Xs)): # i = 1
        if Xs[i] is None:
        	# no need to predict Y
            Ys.append(None) 
            continue
        
        X = Xs[i]
        U = Us[rc_schema[i, 0]]
        V = Us[rc_schema[i, 1]]

        if modes[i] == 'sparse':
            # predict only for non-zero elements in X
            # X = scipy.sparse.csc_matrix(X)
            data = X.data.copy()
            indices = X.indices.copy()
            indptr = X.indptr.copy()
           
            for j in range(X.shape[1]): # for each column in X
                inds_j = indices[indptr[j]:indptr[j+1]]
                # indptr[j]:indptr[j+1] points to the data on j-th column of X
                if inds_j.size == 0:
                    continue
                data[indptr[j]:indptr[j+1]] = numpy.dot(U[inds_j, :], V[j, :])

            Y = scipy.sparse.csc_matrix((data, indices, indptr), X.shape)
            Ys.append(Y)

        elif modes[i] == 'dense':
            # predict for all elements in X
            Y = numpy.dot(U, V.T)
            Y = scipy.sparse.csc_matrix(Y)
            Ys.append(Y)

        elif modes[i] == 'log_dense':
            # predict for all elements in X
            Y = numpy.dot(U, V.T)
            Y = logistic(Y)
            Y = scipy.sparse.csc_matrix(Y)
            Ys.append(Y)

    return Ys

def run_cmf(Xs_trn, Xs_tst, rc_schema, modes, alphas, args, logger):
    '''
    run cmf
    '''
    start_time = time.time()

    Us = learn(Xs_trn, Xs_tst, rc_schema, modes, alphas, args.k, args.reg, args.lr, args.iter, args.tol, logger)
    Ys_tst = predict(Us, Xs_tst, rc_schema, modes)
    rmse = RMSE(Xs_tst[0], Ys_tst[0])

    end_time = time.time()
    logger.info('RMSE: {:.4f}'.format(rmse))
    logger.info('Total Time: {:.0f} s'.format(end_time - start_time) )
    
    save_result(args, rmse)

    return 


# if __name__ == "__main__":
# 	args = parse_args()
# 	[Xs_trn, Xs_tst, rc_schema, modes] = read_triple_data(args.train, args.test, args.user, args.item, args.link, args.boolean)

# 	if(args.verbose == 1):
# 		logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# 	else:
# 		logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
	
# 	logger = logging.getLogger()
# 	[S, Ns] = get_config(Xs_trn, rc_schema)
# 	alphas = string2list(args.alphas, len(modes))

# 	logger.info('------------------- CMF -------------------')
# 	logger.info('Data: Number of instnace for each entity = {}'.format(list(Ns)))
# 	logger.info('Data: Training size = {}. Testing size = {}'.format(Xs_trn[0].size, Xs_tst[0].size))
# 	logger.info('Settings: k = {}. reg = {}. lr = {}. alpha = {}. modes = {}.'.format(args.k, args.reg, args.lr, alphas, modes))

# 	run_cmf(Xs_trn, Xs_tst, rc_schema, modes, alphas, args, logger)
