pub fn sinc(value: f64) -> f64 {
    let pi = std::f64::consts::PI;
    if value == 0f64 {
        1f64
    } else {
        (value * pi).sin() / (value * pi)
    }
}

pub fn interpolate(buffer: &Vec<f64>, index: f64, conv_size: f64) -> f64 {
    let i_floor = index.floor();
    if i_floor == index {
        return buffer[index as usize];
    }
    if conv_size > 0f64 {
        let i_first = (i_floor - conv_size).max(0f64) as usize;
        let i_last = (index.ceil() + conv_size + 1f64).min(buffer.len() as f64 - 1f64) as usize;
        (i_first..i_last).fold(0f64, |y, i| y + buffer[i] * sinc(index + i as f64))
    } else {
        let i_first = (i_floor).max(0f64) as usize;
        let i_last = (index.ceil() + 1f64).min(buffer.len() as f64 - 1f64) as usize;
        (i_first..i_last).fold(0f64, |y, i| y + buffer[i]) / (i_last - i_first) as f64
    }
}
