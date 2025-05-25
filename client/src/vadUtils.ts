export type VADDimension = 'Valence' | 'Arousal' | 'Dominance';

export function getVADDescription(dimension: VADDimension, value: number | null | undefined): string {
    if (value === null || typeof value === 'undefined' || isNaN(value)) {
        return '';
    }

    const val = Math.max(0, Math.min(1, value));

    switch (dimension) {
        case 'Valence':
            if (val < 0.2) return 'Very Unpleasant';
            if (val < 0.4) return 'Unpleasant';
            if (val <= 0.6) return 'Neutral';
            if (val < 0.8) return 'Pleasant';
            return 'Very Pleasant';
        case 'Arousal':
            if (val < 0.2) return 'Very Calm';
            if (val < 0.4) return 'Calm';
            if (val <= 0.6) return 'Neutral';
            if (val < 0.8) return 'Activated';
            return 'Very Activated';
        case 'Dominance':
            if (val < 0.2) return 'Very Controlled';
            if (val < 0.4) return 'Controlled';
            if (val <= 0.6) return 'Neutral';
            if (val < 0.8) return 'In Control';
            return 'Very In Control';
        default:
            if (dimension.toUpperCase() === 'V') return getVADDescription('Valence', value);
            if (dimension.toUpperCase() === 'A') return getVADDescription('Arousal', value);
            if (dimension.toUpperCase() === 'D') return getVADDescription('Dominance', value);
            return '';
    }
}