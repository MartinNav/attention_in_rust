use std::usize;

use ndarray::{Array3,s};
fn min(a:usize,b:usize)->usize{
    if a>b {
        return b;
    }
    a
}
pub fn flash_attention(q:&Array3<f32>, k: &Array3<f32>, v:&Array3<f32>,block_size:usize)->Option<Array3<f32>> {
    let [batch_size, seq_len, head_dim]=q.shape() else{
        return None;
    };
    let num_blocks = (seq_len+block_size-1);
    let output = Array3::zeros((*batch_size,*seq_len,*head_dim));
    for i in  0..num_blocks {
       for j in 0..num_blocks {
           let q_block = q.slice(s![..,(i*block_size)..(min((i+1)*block_size,*seq_len)),..]);
           let k_block = k.slice(s![..,(j*block_size)..(min((j+1)*block_size,*seq_len)),..]);
           let v_block = v.slice(s![..,(j*block_size)..(min((j+1)*block_size,*seq_len)),..]);

           let s = q_block.dot(k_block.t())/(*head_dim as f32).sqrt();
       } 
    }

    Some(output)
    
}
