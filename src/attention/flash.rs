use std::usize;

use ndarray::{s, Array3};
fn min(a: usize, b: usize) -> usize {
    if a > b {
        return b;
    }
    a
}
// this code was copied from AI (https://gemini.google.com/app/50823bc63f3ec0a5)
// The ndarray does not implement this function
// Maybe I should make pr?
/// This function is not the most efficient but it works
fn dot_3d<
    T: num_traits::identities::Zero + Clone + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
>(
    a: &Array3<T>,
    b: &Array3<T>,
) -> Result<Array3<T>, String> {
    let (am, _, ap) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let (bm, bn, bp) = (b.shape()[0], b.shape()[1], b.shape()[2]);
    if ap != bm {
        return Err(format!("Wrong matrix shape"));
    }
    let mut c = Array3::<T>::zeros((am, bn, bp));
    for i in 0..am {
        for j in 0..bn {
            for k in 0..bp {
                for l in 0..ap {
                    c[[i, j, k]] =
                        c[[i, j, k]].clone() + a[[i, l, k]].clone() * b[[l, j, k]].clone();
                }
            }
        }
    }
    Ok(c)
}

pub fn flash_attention(
    q: &Array3<f32>,
    k: &Array3<f32>,
    v: &Array3<f32>,
    block_size: usize,
) -> Option<Array3<f32>> {
    let [batch_size, seq_len, head_dim] = q.shape() else {
        return None;
    };
    let num_blocks = (seq_len + block_size - 1) / block_size;
    let mut output = Array3::zeros((*batch_size, *seq_len, *head_dim));
    for i in 0..num_blocks {
        for j in 0..num_blocks {
            let q_block = q.slice(s![
                ..,
                (i * block_size)..(min((i + 1) * block_size, *seq_len)),
                ..
            ]);
            let k_block = k.slice(s![
                ..,
                (j * block_size)..(min((j + 1) * block_size, *seq_len)),
                ..
            ]);
            let v_block = v.slice(s![
                ..,
                (j * block_size)..(min((j + 1) * block_size, *seq_len)),
                ..
            ]);

            let s = dot_3d(&q_block.to_owned(), &(k_block.t().to_owned())).ok()?
                / (*head_dim as f32).sqrt();
            // I am not sure about this part of implementation being correct
            let mut p = s.map(|&x| x.exp());
            let sum = s.sum();
            //this line is doing absolute sum not multidimensional one
            p = p.map(|&x| x / sum);

            let o = dot_3d(&p.to_owned(), &v_block.to_owned()).ok()?;
            let mut out_slice = output.slice_mut(s![
                ..,
                (i * block_size)..(min((i + 1) * block_size, *seq_len)),
                ..
            ]);
            out_slice.zip_mut_with(&o, |out_val, &o_val| *out_val += o_val);
        }
    }

    Some(output)
}
