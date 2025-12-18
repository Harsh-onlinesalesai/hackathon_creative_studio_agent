To run:
```bash
cd ..

adk web --port 8000 
```        
Sample Input:
```
- Custom User Prompt:
Client Id: 10065131
We need a fresh, high-energy Spotlight banner for the BigBasket app. The visual should feel like a healthy morning breakfast explosion. The yogurt texture needs to look thick and creamy, not watery.

- Marketplace Name: BigBasket

- Product Details:
    - Product Name: "Epigamia Greek Yogurt - Wild Blueberry"
    - Description: High protein, zero preservatives, real fruit chunks. Thick and creamy texture.
    - Product Image: https://www.bbassets.com/media/uploads/p/l/40075139_5-epigamia-greek-yogurt-blueberry.jpg
      (Note: Using a high-quality placeholder that looks like the product for generation purposes)

- Reference Images:
    - Reference 1: https://thumbs.dreamstime.com/b/blueberry-yogurt-vector-realistic-product-placement-mock-up-fresh-splash-fruits-label-design-d-detailed-illustration-141372061.jpg
      (Description: Splash of milk/yogurt with fresh blueberries flying in the air. Bright lighting. Strength: 85/100)
    - Reference 2: https://cdn.prod.website-files.com/6502a82cff431778b5d82829/65151a4e02d548cbf0ca4133_epigamia_logo_300x200px__FitMaxWzQwMCw0MDBd.webp.webp
      (Description: Epigamia Brand Logo, distinct typography. Strength: 100/100)

- Style & Effects:
    - Backgrounds: Clean white studio backdrop or soft pastel purple gradient.
    - Camera Angles: Eye-level close-up of the cup, macro shot of the fruit.
    - Color Preferences: White, Navy Blue, Blueberry Purple.
    - Effects: Dynamic yogurt splash, floating fruit chunks, soft morning shadows.

- CTA (Call to Action):
    - Primary Headline: "YOUR DAILY PROTEIN KICK"
    - Button Style: Rectangular with rounded corners (Pill).
    - Button Text: "Subscribe & Save"

- Mandatory Elements:
    - The text "Real Fruit" must be visible near the yogurt cup.
    - Brand Logo must be placed on the **Top Right** (To comply with Spotlight guidelines avoiding Top Left).

- Inventory Details:
    - Format: Spotlight Widget
    - Aspect Ratio: 3:2
    - Dimensions: 1080x540
    
- Targeting Details:
    - Audience: Health conscious, Gen Z, Gym goers.
    - Geo: Mumbai, Bangalore, Delhi.

- Campaign Objectives:
    - 1. Brand Consideration (Showcase the texture and ingredients).
```

  React Example:

  import { useState } from 'react';

  function CreativeGenerator() {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const generateCreative = async (formData) => {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8000/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: formData })
        });
        const data = await response.json();
        setResult(data);
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setLoading(false);
      }
    };

    return (
      <div>
        {loading && <p>Generating creative...</p>}
        {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
      </div>
    );
  }

  cURL Test:

  curl -X POST "http://localhost:8000/run" \
    -H "Content-Type: application/json" \
    -d '{
      "input": {
        "custom_user_prompt": "Create a BigBasket Spotlight banner",
        "marketplace_name": "BigBasket",
        "product_name": "Greek Yogurt"
      }
    }'

