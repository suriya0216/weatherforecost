/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Syne"', 'sans-serif'],
        body: ['"Sora"', 'sans-serif'],
      },
      colors: {
        brand: {
          blue: '#2563EB',
          violet: '#7C3AED',
          sky: '#0EA5E9',
          ink: '#0F172A',
          soft: '#F5F8FF',
        },
      },
      boxShadow: {
        glow: '0 24px 60px -24px rgba(37, 99, 235, 0.45)',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-12px)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: 0.55, transform: 'scale(1)' },
          '50%': { opacity: 1, transform: 'scale(1.06)' },
        },
        drift: {
          '0%, 100%': { transform: 'translateX(0px)' },
          '50%': { transform: 'translateX(14px)' },
        },
        reveal: {
          from: { opacity: 0, transform: 'translateY(18px)' },
          to: { opacity: 1, transform: 'translateY(0px)' },
        },
      },
      animation: {
        float: 'float 7s ease-in-out infinite',
        'pulse-soft': 'pulse-soft 6s ease-in-out infinite',
        drift: 'drift 9s ease-in-out infinite',
        reveal: 'reveal 0.7s ease-out both',
      },
    },
  },
  plugins: [],
}
