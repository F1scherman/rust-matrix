mod matrix;

fn main() {
    println!("Hello, world!");

    let mut vector : Vec<Vec<f64>> = Vec::with_capacity(10);
    for i in 0..10 {
        vector.push(Vec::with_capacity(5));
        for j in 0..5 {
            vector[i].push(j as f64);
        }
    }

    matrix::Matrix::from_vector(&vector);

    println!("Success!");
}
