// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__h__
#define __ivqML__Model__NeuralNetwork__FeedForward__h__

#include <initializer_list>
#include <vector>
#include <ivqML/Model/Base.h>
#include <ivqML/Model/NeuralNetwork/ActivationFunctions.h>

namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class FeedForward
        : public ivqML::Model::Base< _TReal, _TNatural >
      {
      public:
        using TReal      = _TReal;
        using TNatural   = _TNatural;
        using Self       = FeedForward;
        using Superclass = ivqML::Model::Base< TReal, TNatural >;
        using TMat       = typename Superclass::TMat;
        using TCol       = typename Superclass::TCol;
        using TRow       = typename Superclass::TRow;
        using TMatMap    = typename Superclass::TMatMap;
        using TCMatMap   = typename Superclass::TCMatMap;
        using TColMap    = typename Superclass::TColMap;
        using TCColMap   = typename Superclass::TCColMap;
        using TRowMap    = typename Superclass::TRowMap;
        using TCRowMap   = typename Superclass::TCRowMap;

        using TFunctions
        =
          ivqML::Model::NeuralNetwork::ActivationFunctions< TReal >;
        using TActivation = typename TFunctions::TFunction;

      public:
        FeedForward( );
        virtual ~FeedForward( ) override;

        void set_input_layer(
          const TNatural& i, const TNatural& o, TActivation a
          );
        void set_input_layer(
          const TNatural& i, const TNatural& o, const std::string& a
          );

        void add_layer( const TNatural& o, TActivation a );
        void add_layer( const TNatural& o, const std::string& a );

        TNatural number_of_layers( ) const;

        const TNatural& input_size( const TNatural& l = 0 ) const;
        const TNatural& output_size( const TNatural& l = 0 ) const;

        TReal& operator[]( std::initializer_list< TNatural > i );
        const TReal& operator[]( std::initializer_list< TNatural > i ) const;

        virtual void init( ) override;

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X ) const;

        /**
         * TODO: This method has no sense in neural networks
         */
        template< class _TX, class _Ty >
        void fit(
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1 = 0, const TReal& L2 = 0
          );

        template< class _TG, class _TX, class _Ty >
        TReal cost_gradient(
          Eigen::EigenBase< _TG >& G,
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1, const TReal& L2
          );

        template< class _TX, class _Ty >
        TReal cost(
          const Eigen::EigenBase< _TX >& X,
          const Eigen::EigenBase< _Ty >& y
          );

      protected:
        void _eval( TReal* Ab, TReal* Zb, const TNatural& M, bool keep_AZ ) const;

        virtual void _to_stream( std::ostream& o ) const override;

      protected:
        std::vector< TNatural >    m_N;
        std::vector< TMatMap >     m_W;
        std::vector< TRowMap >     m_B;
        std::vector< TActivation > m_F;
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __ivqML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
