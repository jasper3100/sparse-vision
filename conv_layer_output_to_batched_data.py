# DRAFT CODE FOR USING TRANSFORMING CONV LAYER OUTPUT TO BATCHED DATA

  # the output of the specified layer of the original model is the input of the SAE
            # if the output has 4 dimensions, we flatten it to 2 dimensions along the scheme: (BS, C, H, W) -> (BS*W*H, C)
            if len(output.shape) == 4:
                sae_input = output.reshape(output.size(0)*output.size(2)*output.size(3), output.size(1))
                if sae_input.size(0) > 200: # if the new batch size is too large we process the data in batches --> this makes the code much slower actually...
                    batch_size = closest_batch_size(sae_input.size(0), self.sae_batch_size)
                    dset = Dset(sae_input)
                    sae_data_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
                else: 
                    sae_data_loader = [sae_input] # 
            else: # if the output has 2 dimensions, we just keep it as it is          
                sae_data_loader = [output]

            self.batch_sae_rec_loss[name] = 0.0
            self.batch_sae_l1_loss[name] = 0.0
            self.batch_sae_loss[name] = 0.0
            self.batch_sae_nrmse_loss[name] = 0.0
            self.batch_sae_rmse_loss[name] = 0.0

            for sae_sub_batch in sae_data_loader:
                sae_input = sae_sub_batch.to(self.device)
                encoder_output, decoder_output, encoder_output_prerelu = sae_model(sae_input) 
                accumulated_decoder_output = torch.cat((accumulated_decoder_output, decoder_output), dim=0) if 'accumulated_decoder_output' in locals() else decoder_output
                rec_loss, l1_loss, nrmse_loss, rmse_loss = self.sae_criterion(encoder_output, decoder_output, sae_input) # the inputs are the targets
                loss = rec_loss + self.sae_lambda_sparse*l1_loss
                self.batch_sae_rec_loss[name] += rec_loss.item()
                self.batch_sae_l1_loss[name] += l1_loss.item()
                self.batch_sae_loss[name] += loss.item()
                self.batch_sae_nrmse_loss[name] += nrmse_loss.item()
                self.batch_sae_rmse_loss[name] += rmse_loss.item()

                if train_sae and name == self.train_sae_layer:
                    self.sae_optimizer.zero_grad()
                    loss.backward()
                    sae_model.make_decoder_weights_and_grad_unit_norm()
                    self.sae_optimizer.step()
                
            self.batch_sae_rec_loss[name] /= len(sae_data_loader)
            self.batch_sae_l1_loss[name] /= len(sae_data_loader)
            self.batch_sae_loss[name] /= len(sae_data_loader)
            self.batch_sae_nrmse_loss[name] /= len(sae_data_loader)
            self.batch_sae_rmse_loss[name] /= len(sae_data_loader)

            
            # DO THIS IN THE BATCH LOOP???
            # store quantities of the encoder output
            #self.compute_and_store_batch_wise_metrics(model_key='sae', output=encoder_output, name=name, output_2 = encoder_output_prerelu)

            # we pass the decoder_output back to the original model
            if len(output.shape) == 4:
                accumulated_decoder_output = accumulated_decoder_output.reshape(output.size(0), output.size(2), output.size(3), output.size(1)).permute(0,3,1,2)
                assert accumulated_decoder_output.shape == output.shape
            output = accumulated_decoder_output
