use std::{process::Output, usize};

use ndarray::{Array3,s};
fn min(a:usize,b:usize)->usize{
    if a>b {
        return b;
    }
    a
}
// this code was copied from AI (https://gemini.google.com/app/50823bc63f3ec0a5)
// The ndarray does not implement this function
// Maybe I should make pr?
fn dot_3d<T:num_traits::identities::Zero + Clone + std::ops::Mul<Output = T>+std::ops::Add<Output=T>>(a:&Array3<T>, b: &Array3<T>)->Result<Array3<T>, String>{
    let (am,an,ap)= (a.shape()[0],a.shape()[1],a.shape()[2]);
    let (bm,bn,bp)= (b.shape()[0],b.shape()[1],b.shape()[2]);
    if ap!=bm {
        return Err(format!("Wrong matrix shape"));
    }
    let mut c = Array3::<T>::zeros((am,bn,bp));
    for i in 0..am {
        for j in 0..bn {
            for k in 0..bp {
                for l in 0..ap {
                    c[[i,j,k]]= c[[i,j,k]] + a[[i,l,k]]*b[[l,j,k]];
                }
            }
        }
        
    }
    Ok(c)
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
