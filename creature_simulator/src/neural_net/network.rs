pub struct NeuralNet {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    w1: Vec<f32>, // hidden_size * input_size
    b1: Vec<f32>, // hidden_size
    w2: Vec<f32>, // output_size * hidden_size
    b2: Vec<f32>, // output_size
}

impl NeuralNet {
    /// Number of genome entries required for the given topology.
    pub fn genome_len(input_size: usize, hidden_size: usize, output_size: usize) -> usize {
        hidden_size * input_size + hidden_size + output_size * hidden_size + output_size
    }

    /// Unpack a genome into weight matrices.
    pub fn from_genome(
        genome: &[f32],
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Result<NeuralNet, String> {
        let expected = Self::genome_len(input_size, hidden_size, output_size);
        if genome.len() != expected {
            return Err(format!(
                "genome length {} does not match expected {} for topology {}->{}->{}",
                genome.len(),
                expected,
                input_size,
                hidden_size,
                output_size
            ));
        }

        let mut cursor = 0;
        let mut take = |count: usize| {
            let slice = genome[cursor..cursor + count].to_vec();
            cursor += count;
            slice
        };

        let w1 = take(hidden_size * input_size);
        let b1 = take(hidden_size);
        let w2 = take(output_size * hidden_size);
        let b2 = take(output_size);

        Ok(NeuralNet {
            input_size,
            hidden_size,
            output_size,
            w1,
            b1,
            w2,
            b2,
        })
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Evaluate the network.
    pub fn apply_network(&self, input: &[f32]) -> Vec<f32> {
        let mut hidden = vec![0.0f32; self.hidden_size];
        for h in 0..self.hidden_size {
            let mut sum = self.b1[h];
            let row = h * self.input_size;
            for i in 0..self.input_size {
                let x = input.get(i).copied().unwrap_or(0.0);
                sum += self.w1[row + i] * x;
            }
            hidden[h] = sum.tanh();
        }

        let mut output = vec![0.0f32; self.output_size];
        for o in 0..self.output_size {
            let mut sum = self.b2[o];
            let row = o * self.hidden_size;
            for h in 0..self.hidden_size {
                sum += self.w2[row + h] * hidden[h];
            }
            output[o] = sum.tanh();
        }
        output
    }
}
