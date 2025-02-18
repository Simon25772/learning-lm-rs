use std::{slice, sync::Arc, vec};
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Clone> Clone for Tensor<T>{
    fn clone(&self) -> Self {
        let data_clone:Box<[T]>=self.data.iter().cloned().collect();
        Tensor {
            data:Arc::new(data_clone),
            shape:self.shape.clone(),
            offset:self.offset,
            length:self.length
        }
    }
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }

    pub fn divide_by_row(&self, n: usize) -> Vec<Tensor<T>> {
        let (row, col) = (self.shape()[0], self.shape()[1]);
        assert!(row % n == 0, "Row count must be divisible by n");
    
        let mut result = Vec::new();
        for i in 0..n {
            result.push(self.slice(i * (row / n) * col, &vec![row / n, col]));
        }
        result
    }

    pub fn divide_by_col(&self, n: usize) -> Vec<Tensor<T>> {
        let (row, col) = (self.shape()[0], self.shape()[1]);
        assert!(col % n == 0, "Column count must be divisible by n");
    
        // 创建一个临时的 vector 来保存分割的列数据
        let mut tmp_vec: Vec<Vec<T>> = vec![Vec::new(); n];
        let data = self.data();
        let w = col / n;
    
        // 将数据分配到临时的列 vector 中
        for i in 0..row {
            for j in 0..col {
                tmp_vec[j / w].push(data[i * col + j]);
            }
        }
    
        // 创建结果中的每个 Tensor
        let mut result = Vec::new();
        for i in 0..n {
            result.push(Tensor::new(tmp_vec[i].clone(), &vec![row, col / n]));
        }
        result
    }


}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
