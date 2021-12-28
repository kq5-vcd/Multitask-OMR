from utils import normalize, resize
import cv2

class PriMuS:
    # Data preprocessor/loader for the model
    gt_element_separator = '-'
    PAD_COLUMN = 0


    def __init__(self, corpus_dirpath, corpus_path, word2int, semantic, 
                 img_height, batch_size, img_channels, train_split=0.5, test_split=0.5):
        corpus_file = open(corpus_path,'r')
        corpus_list = corpus_file.read().splitlines()
        corpus_file.close()
        
        del corpus_file

        # Train and validation split
        random.shuffle(corpus_list) 
        train_idx = int(len(corpus_list) * train_split) 
        test_idx = int(len(corpus_list) * test_split) 

        self.training_list = corpus_list[:train_idx]
        self.validation_list = corpus_list[train_idx:-test_idx]
        self.test_list = corpus_list[-test_idx:]
        
        del train_idx
        del test_idx

        print ('Training with ' + str(len(self.training_list)) + ' ,validating with ' 
               + str(len(self.validation_list)) + ' , and testing with ' + str(len(self.test_list)))
        
        self.semantic = semantic
        self.corpus_dirpath = corpus_dirpath
        
        self.current_idx = 0 #identify current index in list of samples
        self.current_eval_idx = 0
        self.current_test_idx = 0

        # Dictionary
        self.word2int = word2int #map symbols to numeric values

        self.vocabulary_size = len(self.word2int)
        
        self.img_height = img_height
        self.batch_size = batch_size
        self.img_channels = img_channels
        
        self.training_iterations = int(len(self.training_list)/self.batch_size) + 1
        self.eval_iterations = int(len(self.validation_list)/self.batch_size) + 1
        self.test_iterations = int(len(self.test_list)/self.batch_size) + 1
        
    
    def load_data(self, filepath):
        sample_fullpath = self.corpus_dirpath + '/' + filepath + '/' + filepath
        #print(sample_filepath)

        # IMAGE
        image = cv2.imread(sample_fullpath + '.png', False)  # Grayscale is assumed!
        
        image = resize(image, self.img_height)
        image = normalize(image)
        

        # GROUND TRUTH
        if self.semantic:
            sample_full_filepath = sample_fullpath + '.semantic'
        else:
            sample_full_filepath = sample_fullpath + '.agnostic'

        sample_gt_file = open(sample_full_filepath, 'r')
        sample_gt_plain = sample_gt_file.readline().rstrip().split('\t')
        sample_gt_file.close()
        
        del sample_fullpath

        label = [self.word2int[lab] for lab in sample_gt_plain] #label: list of numeric values
        
        del sample_gt_plain
        
        return (image, label)
    
    
    def transform_to_batch(self, images):
        # Extend all images to match the longest in the batch
        image_widths = [img.shape[1] for img in images]
        max_image_width = max(image_widths)
        
        del image_widths

        batch_images = np.ones(shape=[len(images),
                                      self.img_channels,
                                      self.img_height,
                                      max_image_width], dtype=np.float32)*self.PAD_COLUMN
        # batch shape: (b, c, h, w)
        
        del max_image_width

        for i, img in enumerate(images):
            batch_images[i, 0, 0:img.shape[0], 0:img.shape[1]] = img
        # shorter images will be padded

        return batch_images

        
    def next_batch(self, phase="train"):
        # Create a batch
        images = [] 
        labels = []

        if phase == "train":
            for _ in range(self.batch_size):
                image, label = self.load_data(self.training_list[self.current_idx])
                images.append(image)
                labels.append(label)

                self.current_idx = (self.current_idx + 1) % len(self.training_list) #increment index, turn back to beginning if overflow
        elif phase == "eval":
            for _ in range(self.batch_size):
                image, label = self.load_data(self.validation_list[self.current_eval_idx])
                images.append(image)
                labels.append(label)
                
                self.current_eval_idx = (self.current_eval_idx + 1) % len(self.validation_list)
        elif phase == "test":
            for _ in range(self.batch_size):
                image, label = self.load_data(self.test_list[self.current_test_idx])
                images.append(image)
                labels.append(label)
            
                self.current_test_idx = (self.current_test_idx + 1) % len(self.test_list)

        # Transform to batch
        batch_images = self.transform_to_batch(images, labels)
        target_lengths = [len(x) for x in labels]
        
        flattened_labels = []
        for label in labels:
            flattened_labels += label
            
        del labels
        
        batch_images = torch.from_numpy(batch_images).to(device)
        labels = torch.tensor(flattened_labels).to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32).to(device)

        return (batch_images, labels, target_lengths)
    

class MTLPriMuS:
    # Data preprocessor/loader for the model
    gt_element_separator = '-'
    PAD_COLUMN = 0


    def __init__(self, corpus_dirpath, corpus_path, word2int_sem, word2int_agn, 
                 img_height, batch_size, img_channels, train_split=0.5, test_split=0.5):
        corpus_file = open(corpus_path,'r')
        corpus_list = corpus_file.read().splitlines()
        corpus_file.close()
        
        del corpus_file

        # Train and validation split
        random.shuffle(corpus_list) 
        train_idx = int(len(corpus_list) * train_split) 
        test_idx = int(len(corpus_list) * test_split) 

        self.training_list = corpus_list[:train_idx]
        self.validation_list = corpus_list[train_idx:-test_idx]
        self.test_list = corpus_list[-test_idx:]
        
        del train_idx
        del test_idx

        print ('Training with ' + str(len(self.training_list)) + ' ,validating with ' 
               + str(len(self.validation_list)) + ' , and testing with ' + str(len(self.test_list)))
        
        self.corpus_dirpath = corpus_dirpath
        
        self.current_idx = 0 #identify current index in list of samples
        self.current_eval_idx = 0
        self.current_test_idx = 0

        # Dictionary
        self.word2int_sem = word2int_sem #map symbols to numeric values
        self.word2int_agn = word2int_agn
        
        self.img_height = img_height
        self.batch_size = batch_size
        self.img_channels = img_channels
        
        self.training_iterations = int(len(self.training_list)/self.batch_size) + 1
        self.eval_iterations = int(len(self.validation_list)/self.batch_size) + 1
        self.test_iterations = int(len(self.test_list)/self.batch_size) + 1
        
    
    def load_data(self, filepath):
        sample_fullpath = self.corpus_dirpath + '/' + filepath + '/' + filepath
        #print(sample_filepath)

        # IMAGE
        image = cv2.imread(sample_fullpath + '.png', False)  # Grayscale is assumed!
        
        image = resize(image, self.img_height)
        image = normalize(image)
        

        # GROUND TRUTH
        sample_gt_file_sem = open(sample_fullpath + '.semantic', 'r')
        sample_gt_plain_sem = sample_gt_file_sem.readline().rstrip().split('\t')
        sample_gt_file_sem.close()
        
        del sample_gt_file_sem
        
        sample_gt_file_agn = open(sample_fullpath + '.agnostic', 'r')
        sample_gt_plain_agn = sample_gt_file_agn.readline().rstrip().split('\t')
        sample_gt_file_agn.close()
        
        del sample_fullpath
        del sample_gt_file_agn

        label_sem = [self.word2int_sem[lab] for lab in sample_gt_plain_sem] #label: list of numeric values
        del sample_gt_plain_sem
        
        label_agn = [self.word2int_agn[lab] for lab in sample_gt_plain_agn] 
        del sample_gt_plain_agn
        
        return (image, label_sem, label_agn)
    
    
    def transform_to_batch(self, images):
        # Extend all images to match the longest in the batch
        image_widths = [img.shape[1] for img in images]
        max_image_width = max(image_widths)
        
        del image_widths

        batch_images = np.ones(shape=[len(images),
                                      self.img_channels,
                                      self.img_height,
                                      max_image_width], dtype=np.float32)*self.PAD_COLUMN
        # batch shape: (b, c, h, w)
        
        del max_image_width

        for i, img in enumerate(images):
            batch_images[i, 0, 0:img.shape[0], 0:img.shape[1]] = img
        # shorter images will be padded

        return batch_images

        
    def next_batch(self, phase="train"):
        # Create a batch
        images = [] 
        labels_sem = []
        labels_agn = []

        if phase == "train":
            for _ in range(self.batch_size):
                image, label_sem, label_agn = self.load_data(self.training_list[self.current_idx])
                images.append(image)
                labels_sem.append(label_sem)
                labels_agn.append(label_agn)

                self.current_idx = (self.current_idx + 1) % len(self.training_list) #increment index, turn back to beginning if overflow
        elif phase == "eval":
            for _ in range(self.batch_size):
                image, label_sem, label_agn = self.load_data(self.validation_list[self.current_eval_idx])
                images.append(image)
                labels_sem.append(label_sem)
                labels_agn.append(label_agn)
                
                self.current_eval_idx = (self.current_eval_idx + 1) % len(self.validation_list)
        elif phase == "test":
            for _ in range(self.batch_size):
                image, label_sem, label_agn = self.load_data(self.test_list[self.current_test_idx])
                images.append(image)
                labels_sem.append(label_sem)
                labels_agn.append(label_agn)
            
                self.current_test_idx = (self.current_test_idx + 1) % len(self.test_list)

        # Transform to batch
        batch_images = self.transform_to_batch(images)
        target_lengths_sem = [len(x) for x in labels_sem]
        target_lengths_agn = [len(x) for x in labels_agn]
        
        flattened_labels_sem = []
        for label in labels_sem:
            flattened_labels_sem += label
        del labels_sem
        
        flattened_labels_agn = []
        for label in labels_agn:
            flattened_labels_agn += label
        del labels_agn
        
        batch_images = torch.from_numpy(batch_images).to(device)
        labels_sem = torch.tensor(flattened_labels_sem).to(device)
        target_lengths_sem = torch.tensor(target_lengths_sem, dtype=torch.int32).to(device)
        labels_agn = torch.tensor(flattened_labels_agn).to(device)
        target_lengths_agn = torch.tensor(target_lengths_agn, dtype=torch.int32).to(device)

        return (batch_images, labels_sem, target_lengths_sem, labels_agn, target_lengths_agn)