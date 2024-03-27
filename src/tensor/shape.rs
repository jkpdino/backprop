pub trait Shape: Sync + Send + 'static {
    const SIZE: usize;
}

#[derive(Clone)]
pub struct Rank1<const A: usize>;

impl<const A: usize> Shape for Rank1<A> {
    const SIZE: usize = A;
}

#[derive(Clone)]
pub struct Rank2<const A: usize, const B: usize>;

impl<const A: usize, const B: usize> Shape for Rank2<A, B> {
    const SIZE: usize = A * B;
}