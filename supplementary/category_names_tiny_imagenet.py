        '''
        # get a list of all category names
        tiny_imagenet_paths = TinyImageNetPaths(root_dir, download=False)
        all_category_names = []
        for _, category_names in tiny_imagenet_paths.nid_to_words.items():
            all_category_names.extend(category_names)
        self.category_names = all_category_names
        '''
        # if somehow the above don't work to get the category names then we could also
        # use the info contained in the pre-trained ResNet50 model (at least that's known to
        # work) but it's not ideal as we don't want to have to load the model just to get the labels
        ''' # something like that
        _, _, self.img_size, self.category_names = load_data_aux(self.dataset_name, 
                                            data_dir=None, 
                                            layer_name=self.layer_name)
        _, self.weights = load_model_aux(self.model_name, 
                                         self.img_size, 
                                         expansion_factor=None)
        score, class_ids = self.classification_results_aux(output)

        category_names = [self.weights.meta["categories"][index] for index in class_ids]
        '''