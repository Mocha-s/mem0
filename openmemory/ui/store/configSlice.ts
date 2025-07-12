import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface LLMConfig {
  model: string;
  temperature: number;
  max_tokens: number;
  api_key?: string;
  ollama_base_url?: string;
}

export interface LLMProvider {
  provider: string;
  config: LLMConfig;
}

export interface EmbedderConfig {
  model: string;
  api_key?: string;
  ollama_base_url?: string;
}

export interface EmbedderProvider {
  provider: string;
  config: EmbedderConfig;
}

export interface Mem0Config {
  llm?: LLMProvider;
  embedder?: EmbedderProvider;
}

export interface OpenMemoryConfig {
  custom_instructions?: string | null;
}

// 新增：Mem0服务配置接口
export interface Mem0ServiceConfig {
  base_url: string;
  api_key?: string;
  timeout: number;
  features: Record<string, FeatureConfig>;
  health_check_interval: number;
}

export interface FeatureConfig {
  enabled: boolean;
  config?: Record<string, any>;
}

// 新增：服务状态接口
export interface ServiceStatus {
  status: 'healthy' | 'unhealthy' | 'unknown';
  last_check: string;
  version?: string;
  error?: string;
}

export interface ConfigState {
  openmemory: OpenMemoryConfig;
  mem0: Mem0Config;
  // 新增：Mem0服务配置
  mem0_service: Mem0ServiceConfig;
  // 新增：服务状态
  service_status: {
    openmemory: ServiceStatus;
    mem0: ServiceStatus;
  };
  status: 'idle' | 'loading' | 'succeeded' | 'failed';
  error: string | null;
}

const initialState: ConfigState = {
  openmemory: {
    custom_instructions: null,
  },
  mem0: {
    llm: {
      provider: 'openai',
      config: {
        model: 'gpt-4o-mini',
        temperature: 0.1,
        max_tokens: 2000,
        api_key: 'env:OPENAI_API_KEY',
      },
    },
    embedder: {
      provider: 'openai',
      config: {
        model: 'text-embedding-3-small',
        api_key: 'env:OPENAI_API_KEY',
      },
    },
  },
  // 新增：默认Mem0服务配置
  mem0_service: {
    base_url: 'http://localhost:8000',
    timeout: 30000,
    health_check_interval: 30000,
    features: {
      'contextual_add': { enabled: true },
      'advanced_search': { enabled: true },
      'criteria_search': { enabled: true },
      'custom_categories': { enabled: true },
      'memory_export': { enabled: true },
      'memory_import': { enabled: true },
      'multimodal': { enabled: false },
      'graph_memory': { enabled: true }
    }
  },
  // 新增：默认服务状态
  service_status: {
    openmemory: {
      status: 'unknown',
      last_check: new Date().toISOString()
    },
    mem0: {
      status: 'unknown',
      last_check: new Date().toISOString()
    }
  },
  status: 'idle',
  error: null,
};

const configSlice = createSlice({
  name: 'config',
  initialState,
  reducers: {
    setConfigLoading: (state) => {
      state.status = 'loading';
      state.error = null;
    },
    setConfigSuccess: (state, action: PayloadAction<{ openmemory?: OpenMemoryConfig; mem0: Mem0Config }>) => {
      if (action.payload.openmemory) {
        state.openmemory = action.payload.openmemory;
      }
      state.mem0 = action.payload.mem0;
      state.status = 'succeeded';
      state.error = null;
    },
    setConfigError: (state, action: PayloadAction<string>) => {
      state.status = 'failed';
      state.error = action.payload;
    },
    updateOpenMemory: (state, action: PayloadAction<OpenMemoryConfig>) => {
      state.openmemory = action.payload;
    },
    updateLLM: (state, action: PayloadAction<LLMProvider>) => {
      state.mem0.llm = action.payload;
    },
    updateEmbedder: (state, action: PayloadAction<EmbedderProvider>) => {
      state.mem0.embedder = action.payload;
    },
    updateMem0Config: (state, action: PayloadAction<Mem0Config>) => {
      state.mem0 = action.payload;
    },
    // 新增：更新Mem0服务配置
    updateMem0ServiceConfig: (state, action: PayloadAction<Mem0ServiceConfig>) => {
      state.mem0_service = action.payload;
    },
    // 新增：更新功能特性配置
    updateFeatureConfig: (state, action: PayloadAction<{ feature: string; config: FeatureConfig }>) => {
      state.mem0_service.features[action.payload.feature] = action.payload.config;
    },
    // 新增：批量更新功能特性
    updateFeaturesConfig: (state, action: PayloadAction<Record<string, FeatureConfig>>) => {
      state.mem0_service.features = { ...state.mem0_service.features, ...action.payload };
    },
    // 新增：更新服务状态
    updateServiceStatus: (state, action: PayloadAction<{ service: 'openmemory' | 'mem0'; status: ServiceStatus }>) => {
      state.service_status[action.payload.service] = action.payload.status;
    },
    // 新增：批量更新服务状态
    updateAllServiceStatus: (state, action: PayloadAction<{ openmemory: ServiceStatus; mem0: ServiceStatus }>) => {
      state.service_status = action.payload;
    },
  },
});

export const {
  setConfigLoading,
  setConfigSuccess,
  setConfigError,
  updateOpenMemory,
  updateLLM,
  updateEmbedder,
  updateMem0Config,
  updateMem0ServiceConfig,
  updateFeatureConfig,
  updateFeaturesConfig,
  updateServiceStatus,
  updateAllServiceStatus,
} = configSlice.actions;

export default configSlice.reducer; 