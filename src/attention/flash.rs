use std::usize;

use ndarray::Array3;

pub fn flash_attention(q:&Array3<f32>, k: &Array3<f32>, v:&Array3<f32>,block_size:usize)->Option<Array3<f32>> {
    let [batch_size, seq_len, head_dim]=q.shape() else{
        return None;
    };
    let num_blocks = (seq_len+block_size-1);
    let output = Array3::zeros((*batch_size,*seq_len,*head_dim));
    for i in  0..num_blocks {
        
    }

    Some(output)
    
}
