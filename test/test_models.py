import unittest
# add parent path
import sys
sys.path.append('./')
from models import GRUNet, MetaGRUNet, compute_loss_meta_gru_net, compute_loss_gru_meta_gru_net, LinearQuadTreeNetwork, FullLinearQuadTreeNetwork, MetaNetwork, Markov1Generator, BaseQuadTreeNetwork
from my_utils import construct_default_quadtree, privtree_clustering, depth_clustering
import numpy as np
import torch
from opacus import PrivacyEngine
from dataset import TrajectoryDataset
from run import train_with_discrete_time
from grid import priv_tree
import torch.nn.functional as F

class Markov1GeneratorTestCase(unittest.TestCase):
	def setUp(self) -> None:
		transition_matrix = torch.tensor([[0,1,0,0], [0,0,1,0], [0,0,0,1]])
		state_to_class = {0:0, 1:1, 2:1, 3:1}
		self.model = Markov1Generator(transition_matrix, state_to_class)

	def test_test(self):

		references = [(0,1,0), (1,1,2)]
		sampled = self.model.make_sample(references, 2)
		self.assertTrue(torch.allclose(torch.tensor(sampled), torch.tensor([[0,1,0],[1,2,0]])))

class GRUNetTestCase(unittest.TestCase):
	def setUp(self) -> None:
		print ("In method:" + self._testMethodName)
		n_locations = 16
		hidden_dim = 30
		batch_size = 10
		n_data = 100
		location_embedding_dim = 10
		n_split = 5
		time_dim = n_split+3

		pad = np.ones((n_data, 1), dtype=int)
		traj1 = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
		traj2 = np.concatenate([np.zeros((n_data, 1)), pad, 2*np.ones((n_data, 1))], axis=1).tolist()
		traj = np.concatenate([traj1, traj2], axis=0).tolist()
		traj_time = [[0, 1, 2]]*n_data*2
		dataset = TrajectoryDataset(traj, traj_time, n_locations, n_split)
		traj_type_dim = len(dataset.label_to_reference)
		self.data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=dataset.make_padded_collate(True))

		self.model = GRUNet(n_locations, location_embedding_dim, time_dim, traj_type_dim, hidden_dim, dataset.reference_to_label)
		
	def tearDown(self):
		print("Out method:" + self._testMethodName)


	def test_make_samples(self):
		# when real_start = True
		references = [(0,1,0), (13,1,2)]
		sampled = self.model.make_sample(references, 2)
		trajs = sampled[0]
		self.assertEqual(trajs[0][0], 0)
		self.assertEqual(trajs[0][2], 0)
		self.assertEqual(trajs[1][0], 13)
		self.assertNotEqual(trajs[1][2], 13)

		# when real_start = False
		references = [(-1,1,0), (-1,1,2)]
		sampled = self.model.make_sample(references, 2)
		trajs = sampled[0]
		print(trajs)


	def test_train(self):
		loss_model = compute_loss_meta_gru_net
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

		# privacy_engine = PrivacyEngine()
		# gru_net, optimizer, data_loader = privacy_engine.make_private(module=gru_net, optimizer=optimizer, data_loader=data_loader, noise_multiplier=1, max_grad_norm=1)
		for i, batch in enumerate(self.data_loader):
			input_locations = batch["input"]
			target_locations = batch["target"]
			references = [tuple(v) for v in batch["reference"]]
			input_times = batch["time"]
			target_times = batch["time_target"]
			
			loss1, loss2, norms = train_with_discrete_time(self.model, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, 1, 1)

		references = [(0,1,0), (0,1,0)]
		sampled = self.model.make_sample(references, 2)

class MetaGRUNetTestCase(unittest.TestCase):
	def setUp(self):
		n_locations = 16
		traj_type_dim = 1
		hidden_dim = 30
		meta_hidden_dim = 40
		memory_dim = 50
		batch_size = 10
		n_data = 100
		location_embedding_dim = 10
		n_split = 5
		time_dim = n_split+3
		multilayer = False
		is_consistent = False

		pad = np.ones((n_data, 1), dtype=int)
		traj = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
		traj_time = [[0, 1, 2, 3]]*n_data
		dataset = TrajectoryDataset(traj, traj_time, n_locations, n_split)
		self.data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=dataset.make_padded_collate(True))

		n_classes = 2

		meta_network = False
		if meta_network:
			meta_network = MetaNetwork(meta_hidden_dim, memory_dim, n_locations, n_classes, "relu")
		else:
			privtree = construct_default_quadtree(2)
		counts = [0]*n_locations
		counts[10] = 1000
		_, privtree = privtree_clustering(counts, theta=10)
		meta_network = FullLinearQuadTreeNetwork(n_locations, memory_dim, meta_hidden_dim, location_embedding_dim, privtree, "relu", multilayer, is_consistent)
		if hasattr(meta_network, "remove_class_to_query"):
			meta_network.remove_class_to_query()

		self.model = MetaGRUNet(meta_network, n_locations, location_embedding_dim, time_dim, traj_type_dim, hidden_dim, dataset.reference_to_label)

	def test_make_samples(self):
		references = [(0,1,0), (13,1,0)]
		sampled = self.model.make_sample(references, 2)
		print(sampled)

	def test_train(self):
		loss_model = compute_loss_meta_gru_net
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

		# privacy_engine = PrivacyEngine()
		# gru_net, optimizer, data_loader = privacy_engine.make_private(module=gru_net, optimizer=optimizer, data_loader=data_loader, noise_multiplier=1, max_grad_norm=1)

		for i, batch in enumerate(self.data_loader):
			input_locations = batch["input"]
			target_locations = batch["target"]
			references = [tuple(v) for v in batch["reference"]]
			input_times = batch["time"]
			target_times = batch["time_target"]
			
			_ = train_with_discrete_time(self.model, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, 1, 1, True)

		references = [(0,1,0), (0,1,0)]
		sampled = self.model.make_sample(references, 2)


	def test_dp_train(self):
		loss_model = compute_loss_meta_gru_net
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

		privacy_engine = PrivacyEngine()
		self.model, optimizer, data_loader = privacy_engine.make_private(module=self.model, optimizer=optimizer, data_loader=self.data_loader, noise_multiplier=1, max_grad_norm=1)

		for i, batch in enumerate(data_loader):
			input_locations = batch["input"]
			target_locations = batch["target"]
			references = [tuple(v) for v in batch["reference"]]
			input_times = batch["time"]
			target_times = batch["time_target"]
			
			_ = train_with_discrete_time(self.model, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, 1, 1)

		references = [(0,1,0), (0,1,0)]
		sampled = self.model._module.make_sample(references, 2)


class TreeNetworkTestCase(unittest.TestCase):
	def setUp(self):
		print("In method:" + self._testMethodName)
		n_locations = 64
		memory_dim = 10
		hidden_dim = 20
		n_classes = 2
		activate = "relu"
		self.n_nodes = 64 + 16 + 4
		self.model = BaseQuadTreeNetwork(n_locations, memory_dim, hidden_dim, n_classes, activate)
		return super().setUp()
	
	def test_to_location_distribution(self):
		self.model.consistent = False
		scores = torch.zeros(2, 2, int(self.n_nodes/4), 4)
		scores[0,0,0,0] = 1
		scores[0,0,1,3] = 1
		scores[0,0,8,0] = 1
		distribution = self.model.to_location_distribution(scores, 0, consistent)
		self.assertTrue(torch.allclose(torch.exp(distribution[0][0][0]), torch.exp(F.log_softmax(scores[0,0,0], dim=0))))
		self.assertAlmostEqual(torch.exp(distribution[1][0][0][5]).item(), 0.15341678261756897)
		self.assertEqual(torch.exp(distribution[2][0][0]).argmax(), 18)


		self.model.consistent = True
		scores = torch.zeros(2, 2, int(self.n_nodes/4), 4)
		scores[0,0,0,0] = 1
		distribution = self.model.to_location_distribution(scores, 0, consistent)
		self.assertEqual(torch.exp(distribution[0][0][0])[0].item(), (torch.exp(distribution[1][0][0][0]) + torch.exp(distribution[1][0][0][1]) + torch.exp(distribution[1][0][0][4]) + torch.exp(distribution[1][0][0][5])).item())

		self.assertAlmostEqual(torch.exp(distribution[2][0][0]).sum().item(), 1, delta=1e-5)
		self.assertAlmostEqual(torch.exp(distribution[1][0][0])[0].item(), torch.exp(distribution[2][0][0])[0].item() + torch.exp(distribution[2][0][0])[1].item() + torch.exp(distribution[2][0][0])[8].item() + torch.exp(distribution[2][0][0])[9].item())

		# 432 -> 759


class FullTreeNetworkTestCase(unittest.TestCase):
	def setUp(self):
		print("In method:" + self._testMethodName)
		n_locations = 64
		n_bins = 6
		memory_dim = 10
		hidden_dim = 20
		location_embedding_dim = 30
		activate = "relu"
		self.n_nodes = 64 + 16 + 4
		_, quad_tree = depth_clustering(n_bins)
		is_consistent = True
		multilayer = False

		self.model = FullLinearQuadTreeNetwork(n_locations, memory_dim, hidden_dim, location_embedding_dim, quad_tree, activate, multilayer, is_consistent)
		self.model.remove_class_to_query()
		return super().setUp()

	def test_to_location_distribution(self):
		print(self.model.training, self.model.pre_training)
		from run import make_targets_of_all_layers
		a = make_targets_of_all_layers(torch.tensor([4]), self.model.tree)
		print(a)
		# training mode and consitent mode
		batch_size = 1
		seq_len = 1
		scores = torch.zeros(batch_size, seq_len, int(self.n_nodes/4), 4)
		scores[0][0][0][2] = 1
		print(scores)

		log_dist = self.model.to_location_distribution(scores, target_depth=2)[0][0]
		print(log_dist.exp())
		self.assertNotEqual(log_dist.exp().sum(), 1)

		# evaluation mode and consitent mode
		self.model.eval()
		log_dist = self.model.to_location_distribution(scores, target_depth=1)[0][0]
		print(log_dist.exp())
		log_dist = self.model.to_location_distribution(scores, target_depth=2)[0][0]
		print(log_dist.exp())
		log_dist = self.model.to_location_distribution(scores, target_depth=3)[0][0]
		print(log_dist.exp())
		self.assertEqual(log_dist.exp().sum().item(), 1)

		# training mode and not consitent mode
		self.model.is_consistent = False
		self.model.train()
		log_dist = self.model.to_location_distribution(scores)[0][0]
		self.assertEqual(log_dist.exp().sum(), 1)

		# evaluation mode and not consitent mode
		self.model.eval()
		log_dist = self.model.to_location_distribution(scores)[0][0]
		self.assertEqual(log_dist.exp().sum(), 1)

	
	def test_location_embedding(self):
		embedding = self.model.location_embedding(torch.tensor([0]))
		ith_state = self.model.root_value(torch.tensor([0]))
		for linear in self.model.linears:
			ith_state = linear(ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)
		self.assertTrue(torch.allclose(self.model.state_to_location_embedding(ith_state[:,0,:]), embedding))

		embedding = self.model.location_embedding(torch.tensor([7]))
		ith_state = self.model.root_value(torch.tensor([0]))
		for linear in self.model.linears:
			ith_state = linear(ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)[:,1,:]
		self.assertTrue(torch.allclose(self.model.state_to_location_embedding(ith_state[0,:]), embedding))
		
		embedding = self.model.location_embedding(torch.tensor([10]))
		ith_state = self.model.root_value(torch.tensor([0]))
		for i, linear in enumerate(self.model.linears):
			ith_state = linear(ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)[:,i,:]
		self.assertTrue(torch.allclose(self.model.state_to_location_embedding(ith_state[0,:]), embedding))

		embedding = self.model.location_embedding(torch.tensor([1]), True)
		ith_state = self.model.root_value(torch.tensor([0]))
		for i, linear in enumerate(self.model.linears):
			ith_state = linear(ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)[:,0,:]
			break
		self.assertTrue(torch.allclose(self.model.state_to_location_embedding(ith_state[0,:]), embedding))

		embedding = self.model.location_embedding(torch.tensor([5]), True)
		ith_state = self.model.root_value(torch.tensor([0]))
		ith_state = self.model.linears[0](ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)[:,0,:]
		ith_state = self.model.linears[1](ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)[:,0,:]
		self.assertTrue(torch.allclose(self.model.state_to_location_embedding(ith_state[0,:]), embedding))

		embedding = self.model.location_embedding(torch.tensor([7]), True)
		ith_state = self.model.root_value(torch.tensor([0]))
		ith_state = self.model.linears[0](ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)[:,0,:]
		ith_state = self.model.linears[1](ith_state).view(ith_state.shape[0], -1, self.model.memory_dim)[:,2,:]
		self.assertTrue(torch.allclose(self.model.state_to_location_embedding(ith_state[0,:]), embedding, atol=1e-8))


	def test_hidden_to_query(self):
		node1_embedding = self.model.location_embedding(torch.tensor([5]), True)
		query_ = self.model.class_to_query(node1_embedding)
		hidden = torch.zeros(1, 1, 16)
		hidden[0,0,0] = 1
		query = self.model.hidden_to_query(hidden)
		self.assertTrue(torch.allclose(query_, query))

		hidden = torch.zeros(1, 1, 16)
		hidden[0,0,0] = 0.5
		hidden[0,0,3] = 0.5

		node_embedding = 0.5 * self.model.location_embedding(torch.tensor([5]), True) + 0.5 * self.model.location_embedding(torch.tensor([8]), True)
		query_ = self.model.class_to_query(node_embedding)
		query = self.model.hidden_to_query(hidden)
		self.assertTrue(torch.allclose(query_, query))


# class MetaNetworkTestCase(unittest.TestCase):
#   def __init__(self, *args, **kwargs):
#     super(MetaNetworkTestCase, self).__init__(*args, **kwargs)
#     n_bins = 6
#     self.n_locations = (n_bins+2)**2
#     self.memory_dim = 100
#     # self.memory_hidden_dim = 32
#     self.hidden_dim = 256
#     self.location_embedding_dim = 64
#     n_classes = 2
#     location_to_class = {i:0 for i in range(self.n_locations)}
#     location_to_class[5] = 1

#     tree = construct_default_quadtree(n_bins)
#     privtree = construct_default_quadtree(n_bins)
#     counts = [0]*self.n_locations
#     counts[10] = 1000


#     # privtree.register_count(counts)
#     # priv_tree(privtree, theta=10)
#     # self.privtree = privtree
#     _, privtree = privtree_clustering(counts, theta=10)


#     # self.model = MetaNetwork(self.hidden_dim, self.memory_dim, self.n_locations, n_classes, "relu")
#     self.model = FullLinearQuadTreeNetwork(self.n_locations, self.memory_dim, self.hidden_dim, self.location_embedding_dim, privtree, "relu")

#     # self.model = MetaAttentionNetworkDirect(self.memory_dim, self.hidden_dim, self.n_locations, n_classes)
#     # self.model = LinearQuadTreeNetwork(tree, self.memory_dim, self.hidden_dim, n_classes, "relu")
#     # self.model = LinearQuadTreeNetwork2(tree, self.memory_dim, n_classes, "relu")
#     # self.model = LinearQuadTreeNetwork3(tree, self.memory_dim, n_classes, "relu")
#     # self.model = FullLinearQuadTreeNetwork(tree, self.memory_dim, self.hidden_dim, self.location_embedding_dim, privtree, "relu")
#     # print(self.model)
#     # self.model = TConvQuadTreeNetwork(tree, self.memory_dim, n_classes, "relu")
#     # self.model = TConvSeparatedQuadTreeNetwork(tree, self.memory_dim, n_classes, "relu")
#     # self.model = TConvNaiveQuadTreeNetwork(tree, self.memory_dim, n_classes, "relu")
#     # self.model = FullTConvNaiveQuadTreeNetwork(tree, self.memory_dim, privtree, "relu")
#     # self.model = LinearTConvQuadTreeNetwork(tree, self.memory_dim, n_classes)
#     # self.model = FullTConvQuadTreeNetwork(tree, self.memory_dim, n_classes, location_to_class)
#     # self.model = NaiveFullTConvQuadTreeNetwork(tree, self.memory_dim, n_classes, location_to_class)
#     # self.model = NaiveClassFullTConvQuadTreeNetwork(tree, self.memory_dim, n_classes, location_to_class)
#     # self.model = PrivTreeTConvQuadTreeNetwork(tree, self.location_embedding_dim, self.memory_dim, privtree)
#     # self.model = LinearPrivTreeTConvQuadTreeNetwork(tree, self.location_embedding_dim, self.memory_dim, privtree)
#     # self.model = GRUQuadTreeNetwork(tree, self.memory_dim, n_classes, 1)
#     # generator, optimizer, data_loader = privacy_engine.make_private(module=generator, optimizer=optimizer, data_loader=data_loader, noise_multiplier=args.noise_multiplier, max_grad_norm=args.clipping_bound)

#   def test_make_keys(self):
#     # field = torch.zeros(1, self.model.memory_dim)
#     # keys = self.model.make_keys(field)
#     # n_nodes = len(self.model.tree.get_all_nodes())
#     # assert keys.shape[1] == n_nodes - 1, "keys.shape: {}, n_nodes: {}".format(keys.shape, n_nodes)
#     if hasattr(self.model, 'make_keys'):
#       n_nodes = len(self.model.tree.get_all_nodes())
#       hidden = torch.zeros(1, 2, self.model.input_dim)
#       query = self.model.hidden_to_query_(hidden)
#       keys = self.model.make_keys(query.shape)
#       assert keys.shape[1] == 2
#       assert keys.shape[2] == n_nodes - 1, "keys.shape: {}, n_nodes: {}".format(keys.shape, n_nodes)

#       keys = self.model.make_keys(query.shape)
#       if hasattr(self.model, 'tconvs'):
#         query = query.view(1, 2, -1, 1, 1)[:,0]
#         state_depth_1, keys_depth_1 = self.model.extract_key_and_state(self.model.tconvs[0](query))
#         state_depth_2, keys_depth_2 = self.model.extract_key_and_state(self.model.tconvs[1](state_depth_1))
#         key_for_id1 = keys_depth_1[0, :, 0, 0].view(-1)
#         key_for_id2 = keys_depth_1[0, :, 0, 1].view(-1)
#         key_for_id3 = keys_depth_1[0, :, 1, 0].view(-1)
#         key_for_id4 = keys_depth_1[0, :, 1, 1].view(-1)
#         key_for_id5 = keys_depth_2[0, :, 0, 0].view(-1)
#         key_for_id6 = keys_depth_2[0, :, 0, 1].view(-1)
#         key_for_id7 = keys_depth_2[0, :, 1, 0].view(-1)
#         key_for_id8 = keys_depth_2[0, :, 1, 1].view(-1)
#         key_for_id9 = keys_depth_2[0, :, 0, 2].view(-1)
#         key_for_id10 = keys_depth_2[0, :, 0, 3].view(-1)
#         key_for_id11 = keys_depth_2[0, :, 1, 2].view(-1)
#         key_for_id12 = keys_depth_2[0, :, 1, 3].view(-1)
#         key_for_id13 = keys_depth_2[0, :, 2, 0].view(-1)
#         key_for_id14 = keys_depth_2[0, :, 2, 1].view(-1)
#         key_for_id15 = keys_depth_2[0, :, 3, 0].view(-1)
#         key_for_id16 = keys_depth_2[0, :, 3, 1].view(-1)
#         key_for_id17 = keys_depth_2[0, :, 2, 2].view(-1)
#         key_for_id18 = keys_depth_2[0, :, 2, 3].view(-1)
#         key_for_id19 = keys_depth_2[0, :, 3, 2].view(-1)
#         key_for_id20 = keys_depth_2[0, :, 3, 3].view(-1)
#       elif hasattr(self.model, 'linears'):
#         # query = query.view(1, 2, -1, self.model.memory_dim)
#         query = query.view(1, 2, -1, self.model.memory_dim)
#         root_value = self.model.root_value(torch.zeros(*query.shape[:-1]).long())
#         state_depth_1 = self.model.linears[0](root_value).view(1, 2, -1, self.model.memory_dim)
#         state_depth_2 = self.model.linears[1](state_depth_1).view(1, 2, -1, self.model.memory_dim)
#         keys_depth_1 = self.model.state_to_key(state_depth_1)
#         keys_depth_2 = self.model.state_to_key(state_depth_2)
#         key_for_id1 = keys_depth_1[0, 0, 0, :].view(-1)
#         key_for_id2 = keys_depth_1[0, 0, 1, :].view(-1)
#         key_for_id3 = keys_depth_1[0, 0, 2, :].view(-1)
#         key_for_id4 = keys_depth_1[0, 0, 3, :].view(-1)
#         key_for_id5 = keys_depth_2[0, 0, 0, :].view(-1)
#         key_for_id6 = keys_depth_2[0, 0, 1, :].view(-1)
#         key_for_id7 = keys_depth_2[0, 0, 4, :].view(-1)
#         key_for_id8 = keys_depth_2[0, 0, 5, :].view(-1)
#         key_for_id9 = keys_depth_2[0, 0, 2, :].view(-1)
#         key_for_id10 = keys_depth_2[0, 0, 3, :].view(-1)
#         key_for_id11 = keys_depth_2[0, 0, 6, :].view(-1)
#         key_for_id12 = keys_depth_2[0, 0, 7, :].view(-1)
#         key_for_id13 = keys_depth_2[0, 0, 8, :].view(-1)
#         key_for_id14 = keys_depth_2[0, 0, 9, :].view(-1)
#         key_for_id15 = keys_depth_2[0, 0, 12, :].view(-1)
#         key_for_id16 = keys_depth_2[0, 0, 13, :].view(-1)
#         key_for_id17 = keys_depth_2[0, 0, 10, :].view(-1)
#         key_for_id18 = keys_depth_2[0, 0, 11, :].view(-1)
#         key_for_id19 = keys_depth_2[0, 0, 14, :].view(-1)
#         key_for_id20 = keys_depth_2[0, 0, 15, :].view(-1)
#       else:
#         query = field.view(1, -1, self.model.memory_dim)[:,0]
#         places = [torch.nn.functional.one_hot(torch.tensor([place]*1), num_classes=4).float().to(query.device) for place in range(4)]
#         key_for_id1 = self.model.gru_cells[0](places[0], query)
#         key_for_id2 = self.model.gru_cells[0](places[1], query)
#         key_for_id3 = self.model.gru_cells[0](places[2], query)
#         key_for_id4 = self.model.gru_cells[0](places[3], query)
#         key_for_id5 = self.model.gru_cells[0](places[0], key_for_id1.view(1, self.model.memory_dim))
#         key_for_id6 = self.model.gru_cells[0](places[1], key_for_id1.view(1, self.model.memory_dim))
#         key_for_id7 = self.model.gru_cells[0](places[2], key_for_id1.view(1, self.model.memory_dim))
#         key_for_id8 = self.model.gru_cells[0](places[3], key_for_id1.view(1, self.model.memory_dim))
#         key_for_id9 = self.model.gru_cells[0](places[0], key_for_id2.view(1, self.model.memory_dim))
#         key_for_id10 = self.model.gru_cells[0](places[1], key_for_id2.view(1, self.model.memory_dim))
#         key_for_id11 = self.model.gru_cells[0](places[2], key_for_id2.view(1, self.model.memory_dim))
#         key_for_id12 = self.model.gru_cells[0](places[3], key_for_id2.view(1, self.model.memory_dim))
#         key_for_id13 = self.model.gru_cells[0](places[0], key_for_id3.view(1, self.model.memory_dim))
#         key_for_id14 = self.model.gru_cells[0](places[1], key_for_id3.view(1, self.model.memory_dim))
#         key_for_id15 = self.model.gru_cells[0](places[2], key_for_id3.view(1, self.model.memory_dim))
#         key_for_id16 = self.model.gru_cells[0](places[3], key_for_id3.view(1, self.model.memory_dim))
#         key_for_id17 = self.model.gru_cells[0](places[0], key_for_id4.view(1, self.model.memory_dim))
#         key_for_id18 = self.model.gru_cells[0](places[1], key_for_id4.view(1, self.model.memory_dim))
#         key_for_id19 = self.model.gru_cells[0](places[2], key_for_id4.view(1, self.model.memory_dim))
#         key_for_id20 = self.model.gru_cells[0](places[3], key_for_id4.view(1, self.model.memory_dim))
#         key_for_id21 = self.model.gru_cells[0](places[0], key_for_id5.view(1, self.model.memory_dim))
#         key_for_id22 = self.model.gru_cells[0](places[1], key_for_id5.view(1, self.model.memory_dim))
#         key_for_id23 = self.model.gru_cells[0](places[2], key_for_id5.view(1, self.model.memory_dim))
#         key_for_id24 = self.model.gru_cells[0](places[3], key_for_id5.view(1, self.model.memory_dim))
#         key_for_id1 = self.model.hidden_to_key(key_for_id1).view(-1)
#         key_for_id2 = self.model.hidden_to_key(key_for_id2).view(-1)
#         key_for_id3 = self.model.hidden_to_key(key_for_id3).view(-1)
#         key_for_id4 = self.model.hidden_to_key(key_for_id4).view(-1)
#         key_for_id5 = self.model.hidden_to_key(key_for_id5).view(-1)
#         key_for_id6 = self.model.hidden_to_key(key_for_id6).view(-1)
#         key_for_id7 = self.model.hidden_to_key(key_for_id7).view(-1)
#         key_for_id8 = self.model.hidden_to_key(key_for_id8).view(-1)
#         key_for_id9 = self.model.hidden_to_key(key_for_id9).view(-1)
#         key_for_id10 = self.model.hidden_to_key(key_for_id10).view(-1)
#         key_for_id11 = self.model.hidden_to_key(key_for_id11).view(-1)
#         key_for_id12 = self.model.hidden_to_key(key_for_id12).view(-1)
#         key_for_id13 = self.model.hidden_to_key(key_for_id13).view(-1)
#         key_for_id14 = self.model.hidden_to_key(key_for_id14).view(-1)
#         key_for_id15 = self.model.hidden_to_key(key_for_id15).view(-1)
#         key_for_id16 = self.model.hidden_to_key(key_for_id16).view(-1)
#         key_for_id17 = self.model.hidden_to_key(key_for_id17).view(-1)
#         key_for_id18 = self.model.hidden_to_key(key_for_id18).view(-1)
#         key_for_id19 = self.model.hidden_to_key(key_for_id19).view(-1)
#         key_for_id20 = self.model.hidden_to_key(key_for_id20).view(-1)
#         key_for_id21 = self.model.hidden_to_key(key_for_id21).view(-1)
#         key_for_id22 = self.model.hidden_to_key(key_for_id22).view(-1)
#         key_for_id23 = self.model.hidden_to_key(key_for_id23).view(-1)
#         key_for_id24 = self.model.hidden_to_key(key_for_id24).view(-1)

#         field = torch.zeros(1, 2, self.model.memory_dim)
#         target = torch.zeros(1, 2, self.model.tree.max_depth)
#         keys2 = self.model.deconv(field, target=target)
#         assert all(keys2[0,0,0,:] == key_for_id1)
#         assert all(keys2[0,0,1,:] == key_for_id2)
#         assert all(keys2[0,0,2,:] == key_for_id3)
#         assert all(keys2[0,0,3,:] == key_for_id4)
#         assert all(keys2[0,0,4,:] == key_for_id5)
#         assert all(keys2[0,0,5,:] == key_for_id6)
#         assert all(keys2[0,0,6,:] == key_for_id7)
#         assert all(keys2[0,0,7,:] == key_for_id8)
#         assert all(keys2[0,0,8,:] == key_for_id21)
#         assert all(keys2[0,0,9,:] == key_for_id22)
#         assert all(keys2[0,0,10,:] == key_for_id23)
#         assert all(keys2[0,0,11,:] == key_for_id24)

#         field = torch.zeros(1, 2, self.model.memory_dim)
#         target = torch.zeros(1, 2, self.model.tree.max_depth)
#         target[:,:,0] = 1
#         key3 = self.model.deconv(field, target=target)
#         assert all(key3[0,0,0,:] == key_for_id1)
#         assert all(key3[0,0,1,:] == key_for_id2)
#         assert all(key3[0,0,2,:] == key_for_id3)
#         assert all(key3[0,0,3,:] == key_for_id4)
#         assert all(key3[0,0,4,:] == key_for_id9)
#         assert all(key3[0,0,5,:] == key_for_id10)
#         assert all(key3[0,0,6,:] == key_for_id11)
#         assert all(key3[0,0,7,:] == key_for_id12)


#       self.assertTrue(torch.allclose(key_for_id1, keys[0, 0, 0, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id2, keys[0, 0, 1, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id3, keys[0, 0, 2, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id4, keys[0, 0, 3, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id5, keys[0, 0, 4, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id6, keys[0, 0, 5, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id7, keys[0, 0, 6, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id8, keys[0, 0, 7, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id9, keys[0, 0, 8, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id10, keys[0, 0, 9, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id11, keys[0, 0, 10, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id12, keys[0, 0, 11, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id13, keys[0, 0, 12, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id14, keys[0, 0, 13, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id15, keys[0, 0, 14, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id16, keys[0, 0, 15, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id17, keys[0, 0, 16, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id18, keys[0, 0, 17, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id19, keys[0, 0, 18, :].view(-1)))
#       self.assertTrue(torch.allclose(key_for_id20, keys[0, 0, 19, :].view(-1)))


#   def test_location_embedding(self):
#     # if type(self.model) != FullTConvQuadTreeNetwork:
#     #   return
#     if hasattr(self.model, 'location_embedding'):
#       location = torch.tensor([0])
#       embedding = self.model.location_embedding(location)
#       self.assertEqual(embedding.shape, (1, self.location_embedding_dim))

#       states = self.model.make_states([1,1,1]).view(1, -1, self.memory_dim)
#       location_embeddings = self.model.state_to_location_embedding(states)[0]
#       self.assertEqual(location_embeddings.shape[-1], self.location_embedding_dim)
      
#       self.assertTrue(torch.allclose(location_embeddings[20], embedding.view(-1), atol=1e-4))

#       location = torch.tensor([21])
#       embedding = self.model.location_embedding(location)
#       self.assertTrue(torch.allclose(location_embeddings[41], embedding.view(-1), atol=1e-4))

#       location = torch.tensor([0])
#       embedding = self.model.location_embedding(location, is_node_id=True)
#       self.assertTrue(torch.allclose(location_embeddings[0], embedding.view(-1), atol=1e-4))

#       location = torch.tensor([15])
#       embedding = self.model.location_embedding(location, is_node_id=True)
#       self.assertTrue(torch.allclose(location_embeddings[15], embedding.view(-1), atol=1e-4))

#       # print(self.model.hidden_ids)
#       # print(self.privtree.get_leaf_ids_in_tree(self.model.tree))
#       # print([node.state_list for node in self.privtree.get_leafs()])
#       # location = torch.tensor([0])
#       # embedding = self.model.location_embedding(location)

#       # root_value = self.model.root_value
#       # field = root_value.repeat(1, 1).view(1, int(self.memory_dim/2), 1, 1)
#       # keys_depth1 = self.model.location_embedding_tconvs[0](field)
#       # keys_depth2 = self.model.location_embedding_tconvs[1](keys_depth1)

#       # key_for_location0 = keys_depth2[0, :, 0, 0].view(-1)
#       # assert all(key_for_location0 == embedding.view(-1)), "key_for_location0: {}, embedding: {}".format(key_for_location0, embedding.view(-1))

#       # location = torch.tensor([1, 2, 5])
#       # embedding = self.model.location_embedding(location)

#       # root_value = self.model.root_value
#       # field = root_value.repeat(3, 1).view(3, int(self.memory_dim/2), 1, 1)
#       # keys_depth1 = self.model.location_embedding_tconvs[0](field)
#       # keys_depth2 = self.model.location_embedding_tconvs[1](keys_depth1)

#       # key_for_location1 = keys_depth2[0, :, 0, 1].view(-1)
#       # key_for_location2 = keys_depth2[1, :, 0, 2].view(-1)
#       # key_for_location5 = keys_depth2[2, :, 1, 1].view(-1)
#       # assert all(key_for_location1 == embedding[0].view(-1)), "key_for_location1: {}, embedding: {}".format(key_for_location1, embedding[0].view(-1))
#       # assert all(key_for_location2 == embedding[1].view(-1)), "key_for_location2: {}, embedding: {}".format(key_for_location2, embedding[1].view(-1))
#       # assert all(key_for_location5 == embedding[2].view(-1)), "key_for_location5: {}, embedding: {}".format(key_for_location5, embedding[2].view(-1))
        

#   # def test_set_probs_to_tree(self):

#   #   query = torch.zeros(1, 2, self.model.memory_dim)
#   #   node_size = len(self.model.tree.get_all_nodes())-1
#   #   field = torch.zeros(1, 2, self.model.memory_dim)  
#   #   keys = self.model.deconv(field)
#   #   if type(self.model) == LinearTConvQuadTreeNetwork:
#   #     self.model.set_probs_to_tree(keys)
#   #   else:
#   #     self.model.set_probs_to_tree(query, keys)
#     # # matmal key and field
#     # query = query.view(-1, 1, int(self.memory_dim/2))
#     # keys = keys.view(-1, node_size, int(self.memory_dim/2))
#     # scores = torch.bmm(query, keys.transpose(-2,-1))
#     # print(query.shape, keys.shape, scores.shape)

#   def test_(self):
#     # batch_size * seq_len * num_nodes
#     # when n_bins = 6, n_nodes = 4, 16, 64
#     if hasattr(self.model, 'to_location_distribution'):
#       scores = torch.zeros(1, 2, 4+16+64)
#       scores[0,0,0] = 1e+10
#       scores[0,0,4] = 1e+10
#       scores[0,0,20] = 1e+10
#       scores[0,1,0+3] = 1e+10
#       scores[0,1,4+10] = 1e+10
#       scores[0,1,20+40] = 1e+10
#       scores = scores.view(1,2,-1,4)
#       log_dists = self.model.to_location_distribution(scores, target_depth=0)
#       self.assertEqual(torch.exp(log_dists[0])[0][0][0].item(), 1)
#       self.assertEqual(torch.exp(log_dists[1])[0][0][0].item(), 1)
#       self.assertEqual(torch.exp(log_dists[2])[0][0][0].item(), 1)
#       self.assertEqual(torch.exp(log_dists[0])[0][1][3].item(), 1)
#       self.assertEqual(torch.exp(log_dists[1])[0][1][10].item(), 1)
#       self.assertEqual(torch.exp(log_dists[2])[0][1][40].item(), 1)

#   def test_forward(self):
#     # if type(self.model) == GRUQuadTreeNetwork:
#     # hidden = torch.zeros(3, 2, self.model.input_dim)
#     # log_distribution = self.model(hidden)
#     # depth = self.model.tree.max_depth
#     # assert log_distribution.shape == (3, 2, sum([4**depth_ for depth_ in range(depth)]), 4), f"log_distribution.shape: {format(log_distribution.shape), [3, 2, sum([4**depth_ for depth_ in range(depth)]), 4]}"

#     # hidden = torch.zeros(3, 2, self.model.input_dim)
#     # target = torch.zeros(3, 2, self.model.tree.max_depth)
#     # log_distribution = self.model(hidden, target)
#     # assert log_distribution.shape == (3, 2, self.model.tree.max_depth, 4), "log_distribution.shape: {}".format(log_distribution.shape)
#     # assert torch.exp(log_distribution).sum().item() == 3*2*self.model.tree.max_depth, "sum of distribution: {}".format(torch.exp(log_distribution).sum().item())
    

#     hidden = torch.zeros(3, 2, self.model.input_dim)
#     log_distribution = self.model(hidden)
#     if log_distribution.shape[-1] == 4:
#       log_distribution = self.model.to_location_distribution(log_distribution)
#       assert log_distribution.shape == (3,2,self.n_locations), "log_distribution.shape: {}, (3,2,{})".format(log_distribution.shape, self.n_locations)
#     else:
#       self.assertEqual(log_distribution.shape, (3, 2, self.n_locations))

#     hidden = torch.zeros(3, 2, self.model.input_dim)
#     log_distribution = self.model(hidden)
#     if log_distribution.shape[-1] == 4:
#       depth = self.model.tree.max_depth
#       log_distribution = self.model.to_location_distribution(log_distribution, target_depth=0)
#       assert len(log_distribution) == depth
#       for depth_ in range(depth):
#         assert log_distribution[depth_].shape == (3,2,4**(depth_+1)), f"log_distribution.shape: {log_distribution[depth_].shape}, (3,2,{4**(depth_+1)})"
#     # else:
#     #   hidden = torch.zeros(2, self.model.input_dim)
#     #   log_distribution = self.model(hidden)
#     #   assert log_distribution.shape == (2, self.n_locations)
#     #   # close to 2 
#     #   # assert torch.exp(log_distribution).sum().item() == 2, "sum of distribution: {}".format(torch.exp(log_distribution).sum().item())
#     #   assert abs(torch.exp(log_distribution).sum().item() -2) < 1e-5, "sum of distribution: {}".format(torch.exp(log_distribution).sum().item())

#     #   hidden = torch.zeros(3, 2, int(self.memory_dim))
#     #   log_distribution = self.model(hidden)
#     #   assert log_distribution.shape == (3, 2, self.n_locations)
#     #   assert abs(torch.exp(log_distribution).sum().item() -3*2) < 1e-5, "sum of distribution: {}".format(torch.exp(log_distribution).sum().item())

#   def test_backward(self):
#     # if hasattr(self.model, 'gru_cells'):
#     hidden = torch.zeros(2, self.model.input_dim)
#     log_distribution = self.model(hidden)
#     target = torch.zeros_like(log_distribution)
#     target[..., 0] = 1
#     loss = torch.nn.functional.kl_div(log_distribution, target, reduction='batchmean')
#     loss.backward()

#     hidden = torch.zeros(3, 2, self.model.input_dim)
#     log_distribution = self.model(hidden)
#     target = torch.zeros_like(log_distribution)
#     target[..., 0] = 1
#     loss = torch.nn.functional.kl_div(log_distribution, target, reduction='batchmean')
#     loss.backward()
#     # else:
#     #   hidden = torch.zeros(2, int(self.memory_dim))
#     #   log_distribution = self.model(hidden)
#     #   target = torch.zeros(2, self.n_locations)
#     #   target[0, 0] = 1
#     #   target[1, 15] = 1
#     #   loss = torch.nn.functional.kl_div(log_distribution, target, reduction='batchmean')
#     #   loss.backward()

#     #   hidden = torch.zeros(3, 2, int(self.memory_dim))
#     #   log_distribution = self.model(hidden)
#     #   target = torch.zeros(3, 2, self.n_locations)
#     #   target[0, 0, 0] = 1
#     #   target[0, 1, 1] = 1
#     #   target[1, 0, 15] = 1
#     #   target[1, 1, 2] = 1
#     #   target[2, 0, 10] = 1
#     #   target[2, 1, 15] = 1
#     #   loss = torch.nn.functional.kl_div(log_distribution, target, reduction='batchmean')
#     #   loss.backward()

#   def test_pre_training(self):
#     # if hasattr(self.model, 'gru_cells'):
#     hidden = torch.zeros(100, self.model.n_classes)
#     hidden[50:, 0] = 1
#     hidden[:50, 1] = 1
#     optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
#     dataset = torch.utils.data.TensorDataset(hidden)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

#     for batch in data_loader:
#       log_distribution = self.model(hidden)
#       target = torch.zeros_like(log_distribution)
#       target[..., 0] = 1
#       loss = torch.nn.functional.kl_div(log_distribution, target, reduction='batchmean')
#       loss.backward()
#       optimizer.step()
#       optimizer.zero_grad()
#     # else:
#     #   hidden = torch.zeros(100, self.model.n_classes)
#     #   hidden[50:, 0] = 1
#     #   hidden[:50, 1] = 1
#     #   optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
#     #   dataset = torch.utils.data.TensorDataset(hidden)
#     #   data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

#     #   for batch in data_loader:
#     #     target = torch.zeros(len(batch), self.n_locations)
#     #     target[:, 0] = 1
#     #     log_distribution = self.model(hidden)
#     #     loss = torch.nn.functional.kl_div(log_distribution, target, reduction='batchmean')
#     #     loss.backward()
#     #     optimizer.step()
#     #     optimizer.zero_grad()

#   # def test_dp_training(self):
#   #   self.model.remove_embeddings_query()
#   #   # self.model.remove_location_embedding()
#   #   hidden = torch.zeros(100, int(self.memory_dim))
#   #   privacy_engine = PrivacyEngine()
#   #   optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
#   #   dataset = torch.utils.data.TensorDataset(hidden)
#   #   data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
#   #   model, optimizer, data_loader = privacy_engine.make_private(module=self.model, optimizer=optimizer, data_loader=data_loader, noise_multiplier=1, max_grad_norm=1)

#   #   for batch in data_loader:
#   #     target = torch.zeros(len(batch), self.n_locations)
#   #     target[:, 0] = 1
#   #     log_distribution = model(hidden)
#   #     loss = torch.nn.functional.kl_div(log_distribution, target, reduction='batchmean')
#   #     loss.backward()
#   #     optimizer.step()
#   #     optimizer.zero_grad()
#   def test_integration_of_gru_net(self):
#     if hasattr(self.model, 'remove_class_to_query'):
#       self.model.remove_class_to_query()

#     input_dim = self.n_locations+2
#     output_dim = self.n_locations
#     traj_type_dim = 1
#     n_split = 5
#     n_layer = 1
#     embed_dim = 10
#     batch_size = 10
#     n_data = 100

#     pad = np.ones((n_data, 1), dtype=int)
#     traj = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
#     traj_time = [[0, 800, 1200]]*n_data
#     dataset = TrajectoryDataset(traj, traj_time, self.n_locations, n_split, 1439)
#     data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=dataset.make_padded_collate(True))

#     gru_net = MetaGRUNet(self.model, traj_type_dim, self.location_embedding_dim, self.hidden_dim, n_split+3, self.n_locations, n_layer, dataset.reference_to_label)
#     loss_model = compute_loss_meta_gru_net
#     optimizer = torch.optim.Adam(gru_net.parameters(), lr=1e-3)

#     for i, batch in enumerate(data_loader):
#         input_locations = batch["input"]
#         target_locations = batch["target"]
#         references = [tuple(v) for v in batch["reference"]]
#         input_times = batch["time"]
#         target_times = batch["time_target"]
#         print(references)

#         loss1, loss2, norms = train_with_discrete_time(gru_net, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, 1, 1)
#     print(gru_net.make_sample(dataset.references, self.n_locations, dataset._time_end_idx(), batch_size))


#   def test_count_params(self):

#     input_dim = self.n_locations+2
#     output_dim = self.n_locations
#     traj_type_dim = 1
#     n_split = 5
#     n_layer = 1
#     batch_size = 10
#     n_data = 100

#     pad = np.ones((n_data, 1), dtype=int)
#     traj = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
#     traj_time = [[0, 800, 1200, 1439]]*n_data
#     dataset = TrajectoryDataset(traj, traj_time, self.n_locations, n_split, 1439)
#     data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=dataset.make_padded_collate(True))

#     gru_net = MetaGRUNet(self.model, traj_type_dim, self.location_embedding_dim, self.hidden_dim, n_split+3, self.n_locations, n_layer, dataset.reference_to_label)
#     params = 0
#     for p in gru_net.parameters():
#         if p.requires_grad:
#             params += p.numel()
#     print("num params", params)
            
#   # def test_integration_of_gru_net_with_target(self):
#   #   self.model.remove_embeddings_query()
#   #   # if type(self.model) == GRUQuadTreeNetwork:
#   #   input_dim = self.n_locations+2
#   #   output_dim = self.n_locations
#   #   traj_type_dim = 1
#   #   hidden_dim = self.memory_dim
#   #   n_split = 5
#   #   n_layer = 1
#   #   embed_dim = 10
#   #   batch_size = 10
#   #   n_data = 100

#   #   pad = np.ones((n_data, 1), dtype=int)
#   #   traj = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
#   #   traj_time = [[0, 800, 1200, 1439]]*n_data
#   #   dataset = TrajectoryDataset(traj, traj_time, self.n_locations, n_split, 1439)
#   #   data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=dataset.make_padded_collate(True))

#   #   gru_net = MetaGRUNet(self.model, traj_type_dim, self.location_embedding_dim, hidden_dim, n_split+3, self.n_locations, n_layer, dataset.reference_to_label)
#   #   optimizer = torch.optim.Adam(gru_net.parameters(), lr=1e-3)

#   #   for i, batch in enumerate(data_loader):
#   #       input_locations = batch["input"]
#   #       target_locations = batch["target"]
#   #       references = [tuple(v) for v in batch["reference"]]
#   #       input_times = batch["time"]
#   #       target_times = batch["time_target"]
#   #       target_paths = torch.tensor([self.model.tree.state_to_path(state) for state in target_locations.view(-1)]).view(target_locations.shape[0], target_locations.shape[1], self.model.tree.max_depth)
#   #       # target = torch.zeros(*input_locations.shape[:2], self.model.tree.max_depth)
#   #       output_locations, output_times = gru_net([input_locations, input_times], references, target=target_paths)
#   #       output_locations = output_locations.view(-1, 4)
#   #       target_paths = target_paths.view(-1)
#   #       loss = torch.nn.functional.nll_loss(output_locations, target_paths, ignore_index=4, reduction='mean')
#   #       loss.backward()
#   #       optimizer.step()
#   #       optimizer.zero_grad()
      
#   #   gru_net.make_sample(dataset.references, self.n_locations, dataset._time_end_idx(), batch_size)

#   # def test_integration_of_gru_net_dp_with_target(self):
#   #   self.model.remove_embeddings_query()

#   #   # if type(self.model) == GRUQuadTreeNetwork:
#   #   input_dim = self.n_locations+2
#   #   output_dim = self.n_locations
#   #   traj_type_dim = 1
#   #   hidden_dim = self.memory_dim
#   #   n_split = 5
#   #   n_layer = 1
#   #   embed_dim = 10
#   #   batch_size = 10
#   #   n_data = 100

#   #   pad = np.ones((n_data, 1), dtype=int)
#   #   traj = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
#   #   traj_time = [[0, 800, 1200, 1439]]*n_data
#   #   dataset = TrajectoryDataset(traj, traj_time, self.n_locations, n_split, 1439)
#   #   data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=dataset.make_padded_collate(True))

#   #   gru_net = MetaGRUNet(self.model, traj_type_dim, self.location_embedding_dim, hidden_dim, n_split+3, self.n_locations, n_layer, dataset.reference_to_label)
#   #   loss_model = compute_loss_gru_meta_gru_net
#   #   optimizer = torch.optim.Adam(gru_net.parameters(), lr=1e-3)

#   #   privacy_engine = PrivacyEngine()
#   #   gru_net, optimizer, data_loader = privacy_engine.make_private(module=gru_net, optimizer=optimizer, data_loader=data_loader, noise_multiplier=1, max_grad_norm=1)

#   #   for i, batch in enumerate(data_loader):
#   #       input_locations = batch["input"]
#   #       target_locations = batch["target"]
#   #       references = [tuple(v) for v in batch["reference"]]
#   #       input_times = batch["time"]
#   #       target_times = batch["time_target"]
#   #       target_paths = torch.tensor([self.model.tree.state_to_path(state) for state in target_locations.view(-1)]).view(target_locations.shape[0], target_locations.shape[1], self.model.tree.max_depth)
#   #       # target = torch.zeros(*input_locations.shape[:2], self.model.tree.max_depth)
#   #       output_locations, output_times = gru_net([input_locations, input_times], references, target=target_paths)
#   #       loss_location, loss_time = loss_model(target_paths, target_times, output_locations, output_times, 1, 1)
#   #       loss = loss_location + loss_time
#   #       loss.backward()
#   #       optimizer.step()
#   #       optimizer.zero_grad()

#   def test_integration_of_gru_net_dp(self):
#     if hasattr(self.model, "remove_class_to_query"):
#       self.model.remove_class_to_query()

#     input_dim = self.n_locations+2
#     output_dim = self.n_locations
#     traj_type_dim = 1
#     hidden_dim = self.memory_dim
#     n_split = 5
#     n_layer = 1
#     embed_dim = 10
#     batch_size = 10
#     n_data = 100

#     pad = np.ones((n_data, 1), dtype=int)
#     traj = np.concatenate([np.zeros((n_data, 1)), pad, np.zeros((n_data, 1))], axis=1).tolist()
#     traj_time = [[0, 800, 1200, 1439]]*n_data
#     dataset = TrajectoryDataset(traj, traj_time, self.n_locations, n_split, 1439)
#     data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=batch_size, collate_fn=dataset.make_padded_collate(True))

#     gru_net = MetaGRUNet(self.model, traj_type_dim, self.location_embedding_dim, hidden_dim, n_split+3, self.n_locations, n_layer, dataset.reference_to_label)
#     loss_model = compute_loss_meta_gru_net
#     optimizer = torch.optim.Adam(gru_net.parameters(), lr=1e-3)

#     privacy_engine = PrivacyEngine()
#     gru_net, optimizer, data_loader = privacy_engine.make_private(module=gru_net, optimizer=optimizer, data_loader=data_loader, noise_multiplier=1, max_grad_norm=1)

#     for i, batch in enumerate(data_loader):
#         input_locations = batch["input"]
#         target_locations = batch["target"]
#         references = [tuple(v) for v in batch["reference"]]
#         input_times = batch["time"]
#         target_times = batch["time_target"]
        
#         loss1, loss2, norms = train_with_discrete_time(gru_net, optimizer, loss_model, input_locations, target_locations, input_times, target_times, references, 1, 1)

# # class MetaQuadTreeNetworkTestCase(unittest.TestCase):

# #   def __init__(self, *args, **kwargs):
# #     super(MetaQuadTreeNetworkTestCase, self).__init__(*args, **kwargs)
# #     n_bins = 2
# #     self.n_locations = (n_bins+2)**2
# #     memory_dim = 20
# #     probs = np.zeros((2, self.n_locations))
# #     probs[0, 0] = 1
# #     probs[1, 15] = 1
# #     n_classes = len(probs)

# #     tree = construct_default_quadtree(n_bins)
# #     self.model = MetaQuadTreeNetwork(tree, memory_dim, n_classes)

# #   def test_init(self):
# #     tree = self.model.tree
# #     assert len(tree.get_all_nodes()) == 1 + 4 + 16

# #   def test_forward(self):
# #     query = torch.tensor([[1,0]], dtype=torch.float32)
# #     probs, reached_leafs = self.model(query)

# #   def test_compute_loss_meta_quad_tree_attention_net(self):
     
# #     query = torch.tensor([[1,0]], dtype=torch.float32)
# #     outputs = self.model(query)
# #     target_probs = torch.zeros(1, self.n_locations)
# #     target_probs[0, 0] = 1
# #     loss = compute_loss_meta_quad_tree_attention_net(outputs, target_probs, self.model.tree)

# #   def test_compute_probs_from_tree(self):
# #     leafs = [self.model.tree.get_leafs()[0]]
# #     target_probs = torch.zeros(1, self.n_locations)
# #     target_probs[0, 0] = 1
# #     probs = compute_probs_from_tree(self.model.tree, leafs, target_probs)
# #     assert all(np.array(probs).reshape(-1) == np.array([[1,0,0,0],[1,0,0,0]]).reshape(-1))

# #     leafs = [self.model.tree.get_leafs()[0]]
# #     target_probs = torch.zeros(1, self.n_locations)
# #     target_probs[0, 15] = 1
# #     probs = compute_probs_from_tree(self.model.tree, leafs, target_probs)
# #     assert all(np.array(probs).reshape(-1) == np.array([[0,0,0,1],[1/4,1/4,1/4,1/4]]).reshape(-1))

# #     leafs = [self.model.tree.get_leafs()[0], self.model.tree.get_leafs()[1]]
# #     target_probs = torch.zeros(2, self.n_locations)
# #     target_probs[0, 15] = 1
# #     target_probs[1, 4] = 1/2
# #     target_probs[1, 6] = 1/2
# #     probs = compute_probs_from_tree(self.model.tree, leafs, target_probs)
# #     assert all(np.array(probs).reshape(-1) == np.array([[0.0, 0.0, 0.0, 1.0], [0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).reshape(-1))



# # class MetaQuadTreeNetworkMarginalTestCase(unittest.TestCase):

# #   def __init__(self, *args, **kwargs):
# #     super(MetaQuadTreeNetworkMarginalTestCase, self).__init__(*args, **kwargs)
# #     n_bins = 2
# #     self.n_locations = (n_bins+2)**2
# #     memory_dim = 20
# #     probs = np.zeros((2, self.n_locations))
# #     probs[0, 0] = 1
# #     probs[1, 15] = 1
# #     n_classes = len(probs)

# #     tree = construct_default_quadtree(n_bins)
# #     self.model = MetaQuadTreeNetworkMarginal(tree, memory_dim, n_classes)

# #   def test_init(self):
# #     tree = self.model.tree
# #     assert len(tree.get_all_nodes()) == 1 + 4 + 16

# #   def test_forward(self):
# #     query = torch.tensor([[1,0],[0,1]], dtype=torch.float32)
# #     distribution = self.model(query)
# #     assert distribution.shape == (query.shape[0], self.n_locations)
# #     self.assertAlmostEqual(sum(distribution[0]).item(), 1, msg="sum of distribution: {}".format(sum(distribution[0])), delta=1e-5)


# # class MetaQuadTreeAttentionNetworkTestCase(unittest.TestCase):
  
# #   def __init__(self, *args, **kwargs):
# #     super(MetaQuadTreeAttentionNetworkTestCase, self).__init__(*args, **kwargs)
# #     n_bins = 2
# #     self.n_locations = (n_bins+2)**2
# #     memory_dim = 20
# #     probs = np.zeros((2, self.n_locations))
# #     probs[0, 0] = 1
# #     probs[1, 15] = 1

# #     tree = construct_default_quadtree(n_bins)
# #     self.model = MetaQuadTreeAttentionNetwork(tree, self.n_locations, memory_dim, probs)
    

# #   def test_init(self):
# #     tree = self.model.tree
# #     assert len(tree.get_all_nodes()) == 1 + 4 + 16


# #   def test_update_tree(self):
# #     self.model.update_tree()
# #     tree = self.model.tree
# #     nodes = tree.get_all_nodes()
# #     assert len(nodes) == 1 + 4 + 16

# #     def check():
# #       for node in nodes[1:]:
# #         assert hasattr(node, 'key')
# #         assert hasattr(node, 'value')

# #         if not node.is_leaf():
# #           assert hasattr(node, 'children')
# #           assert len(node.children) == 4
# #           assert all(node.key == torch.stack([child.key for child in node.children]).mean(dim=0)), "node.key: {}, mean children.key: {}".format(node.key, torch.stack([child.key for child in node.children]).mean(dim=0))
# #           assert all(node.value == torch.stack([child.value for child in node.children]).mean(dim=0)), "node.value: {}, mean children.value: {}".format(node.value, torch.stack([child.value for child in node.children]).mean(dim=0))
# #         else:
# #           state = node.state_list[0]
# #           assert all(node.key == self.model.memory[state, :self.model.memory_dim])
# #           assert all(node.value == self.model.memory[state, self.model.memory_dim:])
      
# #     check()
# #     with torch.no_grad():
# #       self.model.memory[0, :self.model.memory_dim] = torch.tensor([1]*self.model.memory_dim) + self.model.memory[0, :self.model.memory_dim]
# #     self.model.update_tree()
# #     check()

# #   def test_forward(self):
# #     query = torch.tensor([[1,0]], dtype=torch.float32)
# #     probs, reached_leafs = self.model(query)

# #   def test_compute_loss_meta_quad_tree_attention_net(self):
     
# #     query = torch.tensor([[1,0]], dtype=torch.float32)
# #     outputs = self.model(query)
# #     target_probs = torch.zeros(1, self.n_locations)
# #     target_probs[0, 0] = 1
# #     loss = compute_loss_meta_quad_tree_attention_net(outputs, target_probs, self.model.tree)

# #   def test_compute_probs_from_tree(self):
# #     leafs = [self.model.tree.get_leafs()[0]]
# #     target_probs = torch.zeros(1, self.n_locations)
# #     target_probs[0, 0] = 1
# #     probs = compute_probs_from_tree(self.model.tree, leafs, target_probs)
# #     assert all(np.array(probs).reshape(-1) == np.array([[1,0,0,0],[1,0,0,0]]).reshape(-1))

# #     leafs = [self.model.tree.get_leafs()[0]]
# #     target_probs = torch.zeros(1, self.n_locations)
# #     target_probs[0, 15] = 1
# #     probs = compute_probs_from_tree(self.model.tree, leafs, target_probs)
# #     assert all(np.array(probs).reshape(-1) == np.array([[0,0,0,1],[1/4,1/4,1/4,1/4]]).reshape(-1))

# #     leafs = [self.model.tree.get_leafs()[0], self.model.tree.get_leafs()[1]]
# #     target_probs = torch.zeros(2, self.n_locations)
# #     target_probs[0, 15] = 1
# #     target_probs[1, 4] = 1/2
# #     target_probs[1, 6] = 1/2
# #     probs = compute_probs_from_tree(self.model.tree, leafs, target_probs)
# #     assert all(np.array(probs).reshape(-1) == np.array([[0.0, 0.0, 0.0, 1.0], [0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).reshape(-1))


if __name__ == "__main__":
    unittest.main()